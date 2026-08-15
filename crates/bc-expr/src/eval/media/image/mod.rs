//! Image-decode evaluation for `Expr::Image` (the `.image` namespace).
//!
//! This is the interpreter *oracle* for image decoding. The JIT cannot compile
//! library-backed decode, so `bc-codegen` marks `Expr::Image` unsupported and
//! falls back here — the two never diverge because there is only this one
//! implementation. Decode runs per row over the whole batch; a row whose bytes
//! are null or fail to decode yields a null result (corrupt inputs don't fail
//! the batch), matching the multimodal source's header-metadata convention.

mod hash;
mod probe;
mod quality;
mod transform;

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, FixedSizeListArray, GenericBinaryArray, Int32Array, Int64Array,
    OffsetSizeTrait, StructArray, UInt8Array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field};

use super::map_rows;
use crate::{ExprError, ImageFunc};

mod reencode;

use reencode::{
    auto_orient, convert, crop_dynamic, encode, exif_orientation, letterbox, resize, thumbnail,
};
pub(crate) use reencode::{Bounds, Output};

/// Evaluate an image function over a Binary or LargeBinary array of encoded image bytes.
///
/// Both offset widths are accepted because a media source stores its payloads as
/// `LargeBinary`: 32-bit offsets cap one array at 2 GB in total, which a batch of
/// ordinary video or high-resolution image files reaches. Accepting only `Binary` made
/// `.image.decode()` fail on exactly the inputs the namespace exists for.
/// The scalar arguments an image function may carry, gathered into one struct.
///
/// They arrived as seven positional parameters, which had already earned an
/// `#[allow(clippy::too_many_arguments)]`; `encode` and `letterbox` pushed it further, at
/// which point a caller swapping `width` and `height` is a silent bug the compiler cannot
/// see. Named fields make each call site read as what it is.
#[derive(Debug, Clone, Copy)]
pub(crate) struct ImageArgs<'a> {
    pub width: Option<i64>,
    pub height: Option<i64>,
    pub mean: Option<&'a [f64]>,
    pub std: Option<&'a [f64]>,
    pub channels_first: bool,
    /// The container every bytes-out op writes; `png` when absent.
    pub format: Option<&'a str>,
    /// `convert` only: the target colour mode.
    pub mode: Option<&'a str>,
    /// Encoder quality for the lossy containers, 1..=100.
    pub quality: Option<i64>,
    /// The single scalar knob the photometric ops take; named per op.
    pub factor: Option<f64>,
    /// `letterbox`/`pad` only: the byte the leftover canvas is filled with.
    pub fill: Option<i64>,
}

pub(crate) fn eval_image(
    func: ImageFunc,
    arr: &ArrayRef,
    args: ImageArgs<'_>,
) -> Result<ArrayRef, ExprError> {
    let norm = Normalization::resolve(func, args.mean, args.std, args.channels_first)?;
    // An all-null column is typed `Null`, not `Binary`; see `widen_null_column`.
    if let Some(nulls) = super::widen_null_column(arr) {
        return eval_image_sized::<i32>(func, &nulls, args, norm);
    }
    match arr.data_type() {
        DataType::Binary => eval_image_sized::<i32>(func, arr, args, norm),
        DataType::LargeBinary => eval_image_sized::<i64>(func, arr, args, norm),
        other => Err(ExprError::ExpectedBinary {
            // Namespaced, because `Decode` alone is a name three namespaces share and the
            // error otherwise reported an image failure without saying it was one.
            func: format!("image.{func:?}"),
            got: other.to_string(),
        }),
    }
}

/// Validated per-channel normalization for `ToTensorF32`: `(pixel/255 - mean) / std`,
/// plus the output layout. `mean`/`std` default to identity (`0`/`1`) so a bare
/// `to_tensor_f32` just scales to `[0, 1]`.
struct Normalization {
    mean: [f32; 3],
    inv_std: [f32; 3],
    channels_first: bool,
}

impl Normalization {
    fn resolve(
        func: ImageFunc,
        mean: Option<&[f64]>,
        std: Option<&[f64]>,
        channels_first: bool,
    ) -> Result<Self, ExprError> {
        // These knobs only make sense for the float tensor; reject them elsewhere rather
        // than silently ignore, so a mistaken `.image.to_tensor(...)` with a mean doesn't
        // look like it worked.
        if !matches!(func, ImageFunc::ToTensorF32)
            && (mean.is_some() || std.is_some() || channels_first)
        {
            return Err(ExprError::InvalidArgument {
                func: format!("{func:?}"),
                reason: "mean/std/channels_first apply only to to_tensor_f32".to_string(),
            });
        }
        let m = Self::three("to_tensor_f32 mean", mean, 0.0)?;
        let s = Self::three("to_tensor_f32 std", std, 1.0)?;
        for v in s {
            if v == 0.0 {
                return Err(ExprError::InvalidArgument {
                    func: "to_tensor_f32".to_string(),
                    reason: "std values must be non-zero".to_string(),
                });
            }
        }
        Ok(Self {
            mean: m,
            inv_std: [1.0 / s[0], 1.0 / s[1], 1.0 / s[2]],
            channels_first,
        })
    }

    /// A length-3 (RGB) parameter, or the identity `default` for all three when absent.
    fn three(what: &str, v: Option<&[f64]>, default: f32) -> Result<[f32; 3], ExprError> {
        match v {
            None => Ok([default; 3]),
            Some(s) if s.len() == 3 => Ok([s[0] as f32, s[1] as f32, s[2] as f32]),
            Some(s) => Err(ExprError::InvalidArgument {
                func: "to_tensor_f32".to_string(),
                reason: format!(
                    "{what} must have 3 values (one per RGB channel), got {}",
                    s.len()
                ),
            }),
        }
    }
}

fn eval_image_sized<O: OffsetSizeTrait>(
    func: ImageFunc,
    arr: &ArrayRef,
    args: ImageArgs<'_>,
    norm: Normalization,
) -> Result<ArrayRef, ExprError> {
    let (width, height) = (args.width, args.height);
    // The match above already established the offset width, so this cannot fail.
    let bytes = arr
        .as_any()
        .downcast_ref::<GenericBinaryArray<O>>()
        .ok_or_else(|| ExprError::ExpectedBinary {
            func: format!("{func:?}"),
            got: arr.data_type().to_string(),
        })?;
    // Every bytes-out op writes the same container, resolved once for the batch so an
    // unknown format name is one plan error rather than n identical per-row failures.
    let out = Output::resolve(func, args.format, args.quality)?;
    match func {
        ImageFunc::Decode => decode_dims(bytes),
        ImageFunc::ToTensor => to_tensor(bytes, width, height),
        ImageFunc::ToTensorF32 => to_tensor_f32(bytes, width, height, &norm),
        ImageFunc::CenterCrop => center_crop(bytes, width, height),
        ImageFunc::ToGrayscale => to_grayscale(bytes, width, height),
        ImageFunc::Resize => resize(bytes, width, height, out),
        ImageFunc::Encode => encode(bytes, out),
        ImageFunc::Convert => convert(bytes, args.mode, out),
        ImageFunc::Dhash => dhash(bytes),
        ImageFunc::Brightness => quality::brightness(bytes),
        ImageFunc::Sharpness => quality::sharpness(bytes),
        ImageFunc::AutoOrient => auto_orient(bytes, out),
        ImageFunc::ExifOrientation => exif_orientation(bytes),
        ImageFunc::Thumbnail => thumbnail(bytes, width, out),
        ImageFunc::Letterbox => letterbox(bytes, args),
        ImageFunc::Pad => transform::pad(bytes, args, out),
        ImageFunc::Rotate
        | ImageFunc::FlipHorizontal
        | ImageFunc::FlipVertical
        | ImageFunc::AdjustBrightness
        | ImageFunc::AdjustContrast
        | ImageFunc::AdjustSaturation
        | ImageFunc::AdjustHue
        | ImageFunc::Blur
        | ImageFunc::Sharpen
        | ImageFunc::Invert
        | ImageFunc::Posterize
        | ImageFunc::Solarize
        | ImageFunc::Equalize
        | ImageFunc::AutoContrast => transform::eval(func, bytes, args.factor, out),
        ImageFunc::Phash => hash::phash(bytes),
        ImageFunc::Ahash => hash::ahash(bytes),
        ImageFunc::Entropy => quality::entropy(bytes),
        ImageFunc::Colorfulness => quality::colorfulness(bytes),
        ImageFunc::MeanColor => quality::mean_color(bytes),
        ImageFunc::IsGrayscale => quality::is_grayscale(bytes),
        ImageFunc::AspectRatio => probe::aspect_ratio(bytes),
        ImageFunc::HasAlpha => probe::has_alpha(bytes),
        ImageFunc::Format => probe::format(bytes),
    }
}

/// Validate and narrow a target dimension to `u32`.
///
/// A plain `as u32` cast silently wraps an out-of-range `i64`: a negative value becomes a
/// ~4-billion dimension (an unbounded allocation that aborts the process), and a value past
/// `u32::MAX` wraps to a small one (a silently wrong output size). Both are rejected with a
/// clear error instead — the dimension is a query parameter, so a bad one is a caller bug to
/// surface, not a crash to suffer or a wrong answer to return.
pub(super) fn dim(func: &str, arg: &'static str, value: Option<i64>) -> Result<u32, ExprError> {
    let value = value.ok_or(ExprError::MissingImageArg {
        func: func.to_string(),
        arg,
    })?;
    u32::try_from(value)
        .ok()
        .filter(|&v| v > 0)
        .ok_or(ExprError::InvalidImageDim {
            func: func.to_string(),
            arg,
            value,
            max: u32::MAX,
        })
}

/// Validate a per-row element count and return it as a `usize`.
///
/// Every tensor-producing image function shares one hazard: the per-row element count is
/// also the `FixedSizeList` element length, an Arrow `i32`, and is driven by the query's
/// width/height parameters. At ~26,755² a `w*h*3` request already exceeds `i32::MAX`,
/// where the later `as i32` cast wraps negative (an invalid array / panic) and the
/// `len * per_row` allocation becomes a multi-gigabyte OOM bomb. Rejecting it here, with
/// `per_row` computed in `u64` so the multiply cannot itself overflow, is the guard all of
/// them need — so it lives once. `unit` names what a row counts (``"bytes"`` / ``"floats"``
/// / ``"pixels"``) for the error message.
pub(super) fn element_len_guard(func: &str, per_row: u64, unit: &str) -> Result<usize, ExprError> {
    if per_row > i32::MAX as u64 {
        return Err(ExprError::InvalidArgument {
            func: func.to_string(),
            reason: format!(
                "{per_row} {unit} per row exceeds the maximum element length of {}",
                i32::MAX
            ),
        });
    }
    Ok(per_row as usize)
}

/// Assemble per-row RGB8/gray buffers into a `FixedSizeList<UInt8>` column.
///
/// The tail every `UInt8` image tensor shares: a straight `per_row`-byte memcpy per row
/// into one contiguous child buffer (an undecodable row — `None` — leaves its slot zeroed
/// and is marked null), then the `FixedSizeListArray`. Serial on purpose: it is a memcpy,
/// cheap next to the parallel decode that produced `rows`.
pub(super) fn assemble_u8_tensor(n: usize, per_row: usize, rows: Vec<Option<Vec<u8>>>) -> ArrayRef {
    let mut values: Vec<u8> = vec![0u8; n * per_row];
    let mut valid: Vec<bool> = Vec::with_capacity(n);
    for (i, row) in rows.into_iter().enumerate() {
        match row {
            Some(buf) => {
                values[i * per_row..(i + 1) * per_row].copy_from_slice(&buf);
                valid.push(true);
            }
            None => valid.push(false),
        }
    }
    let field = Arc::new(Field::new("item", DataType::UInt8, false));
    Arc::new(FixedSizeListArray::new(
        field,
        per_row as i32,
        Arc::new(UInt8Array::from(values)),
        Some(NullBuffer::from(valid)),
    ))
}

/// Rec. 601 luma of one RGB pixel, rounded to nearest.
///
/// The single definition of "grey" in this module: `to_grayscale`, `dhash`, and
/// `convert("L")` all reduce a pixel through these weights, so they cannot disagree.
fn rec601([r, g, b]: [u8; 3]) -> u8 {
    // +500 for round-to-nearest on the /1000 divide; the sum maxes at 255000 < u32.
    ((299 * r as u32 + 587 * g as u32 + 114 * b as u32 + 500) / 1000) as u8
}

/// `decode` → struct `{width, height, channels, mode}` (header read only).
///
/// All four facts come from **one** header read. Daft spends a separate function call on
/// each of `image_width`/`image_height`/`image_channel`/`image_mode`, re-reading the
/// header once per fact; here the reader already has the struct in hand, so the extra two
/// fields are free. Project the one you want with `.struct.field("width")`.
fn decode_dims<O: OffsetSizeTrait>(bytes: &GenericBinaryArray<O>) -> Result<ArrayRef, ExprError> {
    use arrow::array::StringArray;

    // Read each header in parallel (an undecodable/null row → `None`), then unzip into
    // the column buffers. The header read is the cost; the unzip is a cheap memcpy.
    let headers: Vec<Option<(u32, u32, i32, &'static str)>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            None
        } else {
            image_header(bytes.value(i))
        }
    });
    let mut widths: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut heights: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut channels: Vec<i32> = Vec::with_capacity(bytes.len());
    let mut modes: Vec<&'static str> = Vec::with_capacity(bytes.len());
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for header in headers {
        match header {
            Some((w, h, c, mode)) => {
                widths.push(w as i32);
                heights.push(h as i32);
                channels.push(c);
                modes.push(mode);
                valid.push(true);
            }
            None => {
                // A struct's child arrays stay full length; the row's null bit is what
                // marks it absent, so the placeholders here are never read.
                widths.push(0);
                heights.push(0);
                channels.push(0);
                modes.push("");
                valid.push(false);
            }
        }
    }
    let nulls = NullBuffer::from(valid);
    let fields = vec![
        Arc::new(Field::new("width", DataType::Int32, false)),
        Arc::new(Field::new("height", DataType::Int32, false)),
        Arc::new(Field::new("channels", DataType::Int32, false)),
        Arc::new(Field::new("mode", DataType::Utf8, false)),
    ];
    let columns: Vec<ArrayRef> = vec![
        Arc::new(Int32Array::from(widths)),
        Arc::new(Int32Array::from(heights)),
        Arc::new(Int32Array::from(channels)),
        Arc::new(StringArray::from(modes)),
    ];
    let struct_arr = StructArray::new(fields.into(), columns, Some(nulls));
    Ok(Arc::new(struct_arr))
}

/// `to_tensor(w, h)` → `FixedSizeList<UInt8>` of length `w*h*3` (RGB8, resized).
fn to_tensor<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("to_tensor", "width", width)?;
    let h = dim("to_tensor", "height", height)?;
    let per_row = element_len_guard("to_tensor", (w as u64) * (h as u64) * 3, "bytes")?;
    // Decode + resize every row in parallel (this is the training-data hot path:
    // `read.images(decode=True)` lowers to exactly this kernel). Each row yields its
    // own `per_row`-byte RGB8 buffer, or `None` on null/undecodable/wrong-size input.
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        decode_rgb_resized(bytes.value(i), w, h).filter(|buf| buf.len() == per_row)
    });
    Ok(assemble_u8_tensor(bytes.len(), per_row, rows))
}

/// `to_tensor_f32(w, h, mean, std, channels_first)` → `FixedSizeList<Float32>` of length
/// `w*h*3`: decode, resize, scale to `[0, 1]`, apply per-channel `(x - mean) / std`, and
/// lay out HWC (default) or CHW.
///
/// This is the model-ready counterpart to [`to_tensor`]. It exists so a vision pipeline —
/// `read.images(decode=True)` → normalize → model — stays entirely in the engine: the
/// `/255`, per-channel standardization, and channel-first permute that a torch pipeline
/// otherwise does in a per-batch Python UDF all happen here in one pass over the decoded
/// RGB8 buffer, and the output is a canonical fixed-shape-tensor column ready for the
/// zero-copy handoff to the model.
fn to_tensor_f32<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    width: Option<i64>,
    height: Option<i64>,
    norm: &Normalization,
) -> Result<ArrayRef, ExprError> {
    use arrow::array::Float32Array;

    let w = dim("to_tensor_f32", "width", width)?;
    let h = dim("to_tensor_f32", "height", height)?;
    let per_row = element_len_guard("to_tensor_f32", (w as u64) * (h as u64) * 3, "floats")?;
    let hw = (w as usize) * (h as usize);

    // Decode+resize in parallel (the hot path), then normalize into the flat child buffer.
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        decode_rgb_resized(bytes.value(i), w, h).filter(|buf| buf.len() == per_row)
    });
    let mut values: Vec<f32> = vec![0.0; bytes.len() * per_row];
    let mut valid: Vec<bool> = Vec::with_capacity(bytes.len());
    for (i, row) in rows.into_iter().enumerate() {
        match row {
            Some(rgb) => {
                let out = &mut values[i * per_row..(i + 1) * per_row];
                // `rgb` is HWC RGB8. Normalize each channel; write HWC or CHW.
                for p in 0..hw {
                    for c in 0..3 {
                        let x = (rgb[p * 3 + c] as f32) / 255.0;
                        let v = (x - norm.mean[c]) * norm.inv_std[c];
                        let idx = if norm.channels_first {
                            c * hw + p
                        } else {
                            p * 3 + c
                        };
                        out[idx] = v;
                    }
                }
                valid.push(true);
            }
            None => valid.push(false),
        }
    }
    let field = Arc::new(Field::new("item", DataType::Float32, false));
    let arr = FixedSizeListArray::new(
        field,
        per_row as i32,
        Arc::new(Float32Array::from(values)),
        Some(NullBuffer::from(valid)),
    );
    Ok(Arc::new(arr))
}

/// `center_crop(w, h)` → `FixedSizeList<UInt8>` of length `w*h*3` (RGB8, center-cropped).
///
/// Decodes at native resolution and copies the centered `(h, w)` window. When the image is
/// smaller than the crop in a dimension the missing border is left zero (black), matching
/// torchvision `CenterCrop`, so the output is always exactly `w*h*3` bytes.
fn center_crop<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("center_crop", "width", width)? as usize;
    let h = dim("center_crop", "height", height)? as usize;
    let per_row = element_len_guard("center_crop", (w as u64) * (h as u64) * 3, "bytes")?;

    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?.into_rgb8();
        let (sw, sh) = (img.width() as i64, img.height() as i64);
        // Centered top-left of the crop window (may be negative when the image is smaller).
        let x0 = (sw - w as i64) / 2;
        let y0 = (sh - h as i64) / 2;
        let src = img.as_raw(); // row-major RGB8
        let mut out = vec![0u8; per_row];
        for oy in 0..h {
            let sy = y0 + oy as i64;
            if sy < 0 || sy >= sh {
                continue; // padded row
            }
            for ox in 0..w {
                let sx = x0 + ox as i64;
                if sx < 0 || sx >= sw {
                    continue; // padded column
                }
                let s = ((sy * sw + sx) * 3) as usize;
                let d = (oy * w + ox) * 3;
                out[d..d + 3].copy_from_slice(&src[s..s + 3]);
            }
        }
        Some(out)
    });

    Ok(assemble_u8_tensor(bytes.len(), per_row, rows))
}

/// `to_grayscale(w, h)` → `FixedSizeList<UInt8>` of length `w*h` (shape `(h, w, 1)`).
///
/// Decode, resize to `(w, h)`, and reduce each RGB pixel to one Rec.601 luminance byte
/// (`round((299·R + 587·G + 114·B) / 1000)`) — the standard grayscale conversion, matching
/// PIL `convert("L")`. Reuses the SIMD decode→resize path, then a cheap per-pixel reduction.
fn to_grayscale<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    width: Option<i64>,
    height: Option<i64>,
) -> Result<ArrayRef, ExprError> {
    let w = dim("to_grayscale", "width", width)?;
    let h = dim("to_grayscale", "height", height)?;
    let per_row = element_len_guard("to_grayscale", (w as u64) * (h as u64), "pixels")?;

    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        // `decode_rgb_resized` yields `w*h*3` RGB8 bytes; reduce each pixel to one luma byte.
        let rgb = decode_rgb_resized(bytes.value(i), w, h).filter(|b| b.len() == per_row * 3)?;
        let mut gray = vec![0u8; per_row];
        for (p, g) in gray.iter_mut().enumerate() {
            let base = p * 3;
            *g = rec601([rgb[base], rgb[base + 1], rgb[base + 2]]);
        }
        Some(gray)
    });

    Ok(assemble_u8_tensor(bytes.len(), per_row, rows))
}

/// `dhash()` → UInt64, the 64-bit difference hash of each image.
///
/// The image is reduced to a 9x8 grayscale thumbnail and each row's 8 adjacent pixel
/// pairs are compared, giving 8x8 = 64 bits of "is this pixel brighter than the one to
/// its right". Comparing *gradients* rather than absolute values is what makes the hash
/// survive re-encoding, rescaling and brightness shifts while still separating different
/// images — which is exactly the invariance an image-dedup pass needs.
///
/// The 64 bits are returned as **`Int64`, not `UInt64`**, reinterpreted rather than
/// clamped. That is not cosmetic: the FFI boundary normalizes to `i64` and *rejects* a
/// `u64` above `i64::MAX`, so a `UInt64` hash would make every hash with its high bit set
/// — half of them — unusable in the very expression this exists to enable. Sign is
/// irrelevant to the intended use, because XOR and popcount are bit operations:
/// `a.bitwise_xor(b).bit_count()` is the Hamming distance whichever way the bits are
/// read, and a threshold on it is a near-duplicate predicate.
fn dhash<O: OffsetSizeTrait>(bytes: &GenericBinaryArray<O>) -> Result<ArrayRef, ExprError> {
    // 9 wide so that each of the 8 output columns is a comparison between two real
    // neighbouring pixels, rather than wrapping at the edge.
    const W: u32 = 9;
    const H: u32 = 8;

    let hashes: Vec<Option<i64>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let rgb = decode_rgb_resized(bytes.value(i), W, H)?;
        if rgb.len() != (W * H * 3) as usize {
            return None;
        }
        let mut bits: u64 = 0;
        for row in 0..H as usize {
            for col in 0..(W - 1) as usize {
                let left = luma(&rgb, row, col, W as usize);
                let right = luma(&rgb, row, col + 1, W as usize);
                bits = (bits << 1) | u64::from(left > right);
            }
        }
        // Reinterpret, never saturate: the bit pattern is the hash.
        Some(bits as i64)
    });
    Ok(Arc::new(Int64Array::from(hashes)))
}

/// Rec. 601 luma of the pixel at `(row, col)` in a row-major RGB8 buffer `width` wide.
///
/// Integer weights (the standard 299/587/114 per mille) keep the hash bit-for-bit
/// reproducible across platforms — a float dot product would not be, and a perceptual
/// hash that varies by machine cannot be stored or compared across runs.
fn luma(rgb: &[u8], row: usize, col: usize, width: usize) -> u32 {
    let base = (row * width + col) * 3;
    let (r, g, b) = (rgb[base] as u32, rgb[base + 1] as u32, rgb[base + 2] as u32);
    299 * r + 587 * g + 114 * b
}

/// Header facts about an image: dimensions, channel count, and color-mode name.
///
/// One header read yields all four. Daft spends a separate call on each of
/// `image_width`/`image_height`/`image_channel`/`image_mode`, which re-reads the header
/// four times for a struct the reader already had in hand.
fn image_header(data: &[u8]) -> Option<(u32, u32, i32, &'static str)> {
    let reader = image::ImageReader::new(Cursor::new(data))
        .with_guessed_format()
        .ok()?;
    let decoder = reader.into_decoder().ok()?;
    let (w, h) = image::ImageDecoder::dimensions(&decoder);
    let (channels, mode) = color_mode(image::ImageDecoder::color_type(&decoder));
    Some((w, h, channels, mode))
}

/// `(channel count, mode name)` for a decoded color type.
///
/// The mode names follow Pillow's vocabulary (`L`, `LA`, `RGB`, `RGBA`), because that is
/// what a multimodal pipeline's other half speaks and an unfamiliar third spelling would
/// help nobody. Bit depth is deliberately not encoded in the name: a 16-bit RGB image is
/// `RGB` with 3 channels, since a caller branching on the mode cares about the channel
/// layout, and the depth is the decoder's business.
fn color_mode(color: image::ColorType) -> (i32, &'static str) {
    use image::ColorType::{La16, La8, Rgb16, Rgb32F, Rgb8, Rgba16, Rgba32F, Rgba8, L16, L8};
    match color {
        L8 | L16 => (1, "L"),
        La8 | La16 => (2, "LA"),
        Rgb8 | Rgb16 | Rgb32F => (3, "RGB"),
        Rgba8 | Rgba16 | Rgba32F => (4, "RGBA"),
        // `ColorType` is non-exhaustive, so an unknown variant reports its channel count
        // from the decoder rather than guessing a name.
        other => (other.channel_count() as i32, "other"),
    }
}

/// Decode, resize to `(w, h)`, and flatten to RGB8; `None` on any failure.
///
/// Two SIMD-accelerated stages, each the fast option for its job:
///   1. **Decode.** For a JPEG whose source is ≥2× the target, decode at a 1/2·1/4·1/8
///      **DCT scale** ([`decode_jpeg_scaled`]) — the physical-AI case (large camera frame
///      → small model input), where full-res decode wastes most of the pixels and memory
///      bandwidth. Everything else (small JPEGs, PNG/WebP/…) decodes via zune-jpeg / the
///      `image` crate, already SIMD.
///   2. **Resize** to the exact `(w, h)` with `fast_image_resize` (SIMD bilinear — the same
///      tent/linear filter the previous scalar `resize_exact(Triangle)` used).
///
/// Pixels are not bit-identical across decoder/resizer implementations; the multimodal
/// benchmark accounts for this by comparing counts/dims, not raw pixels.
fn decode_rgb_resized(data: &[u8], w: u32, h: u32) -> Option<Vec<u8>> {
    use fast_image_resize as fir;

    // Stage 1: decode to an RGB8 buffer at some `(sw, sh)` ≥ the target.
    let (rgb, sw, sh) = match decode_jpeg_scaled(data, w, h) {
        Some(scaled) => scaled,
        None => {
            let img = image::load_from_memory(data).ok()?;
            let (sw, sh) = (img.width(), img.height());
            (img.into_rgb8().into_raw(), sw, sh)
        }
    };
    if sw == 0 || sh == 0 {
        return None;
    }
    // Stage 2: exact resize (skip when the scaled decode already hit the target size).
    if sw == w && sh == h {
        return Some(rgb);
    }
    let src = fir::images::Image::from_vec_u8(sw, sh, rgb, fir::PixelType::U8x3).ok()?;
    let mut dst = fir::images::Image::new(w, h, fir::PixelType::U8x3);
    let opts = fir::ResizeOptions::new()
        .resize_alg(fir::ResizeAlg::Convolution(fir::FilterType::Bilinear));
    fir::Resizer::new().resize(&src, &mut dst, &opts).ok()?;
    Some(dst.into_vec())
}

/// DCT-scaled JPEG decode: decode at the smallest 1/1·1/2·1/4·1/8 scale whose output is
/// still ≥ `(w, h)`, returning `(rgb8, out_w, out_h)`. `None` (→ caller uses the full
/// decoder) unless the source is a baseline RGB JPEG at least 2× the target in both dims —
/// below that, `jpeg-decoder`'s slower per-pixel decode would lose to zune-jpeg's full one.
fn decode_jpeg_scaled(data: &[u8], w: u32, h: u32) -> Option<(Vec<u8>, u32, u32)> {
    use jpeg_decoder::{Decoder, PixelFormat};

    let mut dec = Decoder::new(Cursor::new(data));
    dec.read_info().ok()?;
    let info = dec.info()?;
    // Only a win when at least one DCT octave can be dropped, and only for RGB (the common
    // 3-channel photo case); grayscale/CMYK fall back so the pixel semantics stay simple.
    if info.pixel_format != PixelFormat::RGB24
        || u32::from(info.width) < w.saturating_mul(2)
        || u32::from(info.height) < h.saturating_mul(2)
    {
        return None;
    }
    // Request the target size; the decoder snaps up to the nearest DCT scale that still
    // covers it, so the subsequent resize only ever downsamples.
    let (out_w, out_h) = dec.scale(w as u16, h as u16).ok()?;
    let pixels = dec.decode().ok()?;
    let (out_w, out_h) = (u32::from(out_w), u32::from(out_h));
    if pixels.len() != (out_w as usize) * (out_h as usize) * 3 {
        return None; // unexpected layout — let the full decoder handle it
    }
    Some((pixels, out_w, out_h))
}

/// Evaluate `ImageCrop` — a crop whose window is four columns rather than four constants.
///
/// Split from [`eval_image`] rather than folded into it because the shapes genuinely
/// differ: every `ImageFunc` reads scalar arguments validated once for the batch, while
/// this one reads four arrays validated per row. Sharing an entry point would mean an
/// `ImageArgs` carrying four `ArrayRef`s that fifteen of the sixteen functions ignore.
pub(crate) fn eval_image_crop(arr: &ArrayRef, bounds: &Bounds<'_>) -> Result<ArrayRef, ExprError> {
    // An all-null column is typed `Null`, not `Binary`; see `media::widen_null_column`.
    if let Some(nulls) = super::widen_null_column(arr) {
        let bytes = nulls
            .as_any()
            .downcast_ref::<GenericBinaryArray<i32>>()
            .expect("widen_null_column builds a Binary array");
        return crop_dynamic(bytes, bounds);
    }
    match arr.data_type() {
        DataType::Binary => crop_dynamic(
            arr.as_any()
                .downcast_ref::<GenericBinaryArray<i32>>()
                .expect("matched Binary"),
            bounds,
        ),
        DataType::LargeBinary => crop_dynamic(
            arr.as_any()
                .downcast_ref::<GenericBinaryArray<i64>>()
                .expect("matched LargeBinary"),
            bounds,
        ),
        other => Err(ExprError::ExpectedBinary {
            func: "image.crop".to_string(),
            got: other.to_string(),
        }),
    }
}

#[cfg(test)]
mod tests {
    // The `reencode` tests live here rather than beside their functions because they share
    // this module's fixtures (`ei`, `args`, `red_png`) and its `eval_image` entry point;
    // duplicating those into a second test module would cost more than the distance does.
    use arrow::array::{BinaryArray, LargeBinaryArray};

    use super::reencode::ENCODE_FORMATS;
    use super::*;

    /// `eval_image` without normalization — the non-`to_tensor_f32` ops ignore it, so
    /// the tests for those stay readable at four arguments.
    fn ei(
        func: ImageFunc,
        arr: &ArrayRef,
        w: Option<i64>,
        h: Option<i64>,
    ) -> Result<ArrayRef, ExprError> {
        eval_image(func, arr, args(w, h))
    }

    /// The default `ImageArgs` for a test: just a width and a height.
    fn args<'a>(w: Option<i64>, h: Option<i64>) -> ImageArgs<'a> {
        ImageArgs {
            width: w,
            height: h,
            mean: None,
            std: None,
            channels_first: false,
            format: None,
            mode: None,
            quality: None,
            factor: None,
            fill: None,
        }
    }

    /// A 2×3 red PNG, encoded once so the test has no I/O.
    fn red_png(width: u32, height: u32) -> Vec<u8> {
        let buf = image::RgbImage::from_pixel(width, height, image::Rgb([255, 0, 0]));
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(buf)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    /// Encode an RGB image as PNG so a test can build one inline.
    fn png_of(img: image::RgbImage) -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    /// A smooth 2-D wave with `cycles` horizontal periods.
    ///
    /// A monotonic ramp would be useless here: its gradient has one sign everywhere, so
    /// every comparison is false and the hash is 0 — indistinguishable from a flat image.
    /// A wave reverses direction `cycles` times per row, giving a rich bit pattern, and
    /// it is low-frequency enough to survive downscaling to 9x8 (which is what makes the
    /// rescaling test meaningful rather than accidental).
    fn wave(w: u32, h: u32, cycles: f32) -> image::RgbImage {
        use std::f32::consts::PI;
        image::RgbImage::from_fn(w, h, |x, y| {
            let fx = (x as f32 / w as f32) * cycles * 2.0 * PI;
            let fy = (y as f32 / h as f32) * 2.0 * PI;
            let v = (128.0 + 120.0 * (fx.sin() * fy.cos())).clamp(0.0, 255.0) as u8;
            image::Rgb([v, v, v])
        })
    }

    fn hash_of(png: &[u8]) -> Option<i64> {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(png)]));
        let out = ei(ImageFunc::Dhash, &arr, None, None).unwrap();
        let a = out.as_any().downcast_ref::<Int64Array>().unwrap();
        (!a.is_null(0)).then(|| a.value(0))
    }

    /// The property the whole feature rests on: the same image hashes the same.
    #[test]
    fn dhash_is_deterministic() {
        let png = png_of(wave(64, 64, 3.0));
        let h = hash_of(&png);
        assert_eq!(h, hash_of(&png));
        // Guard the guard: a hash of 0 would make the equality above vacuous, and 0 is
        // exactly what a degenerate test image produces.
        assert_ne!(h, Some(0), "test image has no usable gradient structure");
    }

    /// Rescaling must not change the hash much — that is what makes it *perceptual*
    /// rather than a checksum, and what lets it match a thumbnail to its original.
    #[test]
    fn dhash_survives_rescaling() {
        let big = hash_of(&png_of(wave(256, 256, 3.0))).unwrap();
        let small = hash_of(&png_of(wave(64, 64, 3.0))).unwrap();
        let distance = (big ^ small).count_ones();
        assert!(distance <= 8, "rescaled image moved {distance} bits");
    }

    /// ...while a visibly different image must be far away, or the hash separates nothing.
    #[test]
    fn dhash_separates_different_images() {
        let three = hash_of(&png_of(wave(64, 64, 3.0))).unwrap();
        let seven = hash_of(&png_of(wave(64, 64, 7.0))).unwrap();
        let distance = (three ^ seven).count_ones();
        assert!(
            distance >= 16,
            "different images only {distance} bits apart"
        );
    }

    /// Null and undecodable rows yield null rather than failing the batch, matching
    /// every other media kernel.
    #[test]
    fn dhash_nulls_and_garbage_yield_null() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(png_of(wave(32, 32, 3.0)).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = ei(ImageFunc::Dhash, &arr, None, None).unwrap();
        let a = out.as_any().downcast_ref::<Int64Array>().unwrap();
        assert!(!a.is_null(0));
        assert!(a.is_null(1));
        assert!(a.is_null(2), "undecodable bytes must be null, not an error");
    }

    /// A flat image has no gradients anywhere, so every comparison is false.
    #[test]
    fn dhash_of_a_flat_image_is_zero() {
        let flat = image::RgbImage::from_pixel(32, 32, image::Rgb([128, 128, 128]));
        assert_eq!(hash_of(&png_of(flat)), Some(0));
    }

    /// `LargeBinary` must decode identically to `Binary`.
    ///
    /// A media source stores payloads as `LargeBinary` (32-bit offsets cap an array at
    /// 2 GB *total*, which a batch of images or clips reaches), so this is the layout the
    /// `.image` namespace actually sees in production. Accepting only `Binary` made every
    /// one of these calls fail at run time, and nothing in the Rust tests noticed because
    /// they all built `BinaryArray`.
    #[test]
    fn large_binary_matches_binary_for_every_image_func() {
        let png = red_png(4, 6);
        let rows = vec![Some(png.as_slice()), None];
        let narrow: ArrayRef = Arc::new(BinaryArray::from(rows.clone()));
        let wide: ArrayRef = Arc::new(LargeBinaryArray::from(rows));

        for func in [
            ImageFunc::Decode,
            ImageFunc::ToTensor,
            ImageFunc::Resize,
            ImageFunc::Dhash,
        ] {
            let (w, h) = (Some(2), Some(3));
            let from_narrow = ei(func, &narrow, w, h).unwrap();
            let from_wide = ei(func, &wide, w, h).unwrap();
            assert_eq!(
                from_narrow.as_ref(),
                from_wide.as_ref(),
                "{func:?} disagreed between Binary and LargeBinary"
            );
        }
    }

    /// A type that is neither Binary nor LargeBinary still reports a clear error.
    #[test]
    fn a_non_binary_argument_is_rejected() {
        let arr: ArrayRef = Arc::new(Int32Array::from(vec![1, 2, 3]));
        let err = ei(ImageFunc::Decode, &arr, None, None).unwrap_err();
        assert!(format!("{err}").contains("Int32"), "{err}");
    }

    /// Manual throughput check (ignored). Run:
    ///   RAYON_NUM_THREADS=1 cargo test -p bc-expr to_tensor_bench -- --ignored --nocapture
    ///   cargo test -p bc-expr to_tensor_bench -- --ignored --nocapture
    #[test]
    #[ignore]
    fn to_tensor_bench() {
        use std::time::Instant;
        let n = 4000usize;
        let blobs: Vec<Vec<u8>> = (0..n)
            .map(|i| {
                // A noisy image so JPEG decode does real work (not a trivial solid fill).
                let mut img = image::RgbImage::new(256, 256);
                for (x, y, p) in img.enumerate_pixels_mut() {
                    let v = ((x * 7 + y * 13 + (i as u32) * 31) % 256) as u8;
                    *p = image::Rgb([v, v.wrapping_add(80), v.wrapping_add(160)]);
                }
                let mut out = Cursor::new(Vec::new());
                image::DynamicImage::ImageRgb8(img)
                    .write_to(&mut out, image::ImageFormat::Jpeg)
                    .unwrap();
                out.into_inner()
            })
            .collect();
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(
            blobs.iter().map(|b| Some(b.as_slice())),
        ));
        let _ = ei(ImageFunc::ToTensor, &arr, Some(224), Some(224)).unwrap(); // warm
        let t = Instant::now();
        let out = ei(ImageFunc::ToTensor, &arr, Some(224), Some(224)).unwrap();
        let dt = t.elapsed().as_secs_f64();
        assert_eq!(out.len(), n);
        println!(
            "rayon_threads={}  to_tensor {n} imgs: {:.1} ms ({:.0} img/s)",
            rayon::current_num_threads(),
            dt * 1000.0,
            n as f64 / dt
        );
    }

    #[test]
    fn decode_reads_dimensions() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(2, 3).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = ei(ImageFunc::Decode, &arr, None, None).unwrap();
        let s = out.as_any().downcast_ref::<StructArray>().unwrap();
        let w = s.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
        let h = s.column(1).as_any().downcast_ref::<Int32Array>().unwrap();
        assert!(s.is_valid(0) && w.value(0) == 2 && h.value(0) == 3);
        assert!(s.is_null(1)); // null bytes → null struct
        assert!(s.is_null(2)); // undecodable bytes → null struct
    }

    #[test]
    fn crop_takes_the_named_window_not_the_middle() {
        use image::{Rgb, RgbImage};
        // An 8x8 image, left half red and right half green: a crop at x=0 must be all
        // red, and one at x=4 all green. A centered crop could not tell them apart, which
        // is the whole reason `crop` exists beside `center_crop`.
        let mut img = RgbImage::new(8, 8);
        for y in 0..8 {
            for x in 0..8 {
                img.put_pixel(
                    x,
                    y,
                    if x < 4 {
                        Rgb([255, 0, 0])
                    } else {
                        Rgb([0, 255, 0])
                    },
                );
            }
        }
        let mut src = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut src, image::ImageFormat::Png)
            .unwrap();
        let src = src.into_inner();
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(src.as_slice())]));

        for (x, want) in [(0i64, [255u8, 0, 0]), (4, [0, 255, 0])] {
            let out = crop_at(&arr, &[x], &[0], &[4], &[8]);
            let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
            let got = image::load_from_memory(b.value(0)).unwrap().into_rgb8();
            assert_eq!((got.width(), got.height()), (4, 8), "x={x}");
            assert_eq!(got.get_pixel(0, 0).0, want, "x={x}");
        }
    }

    /// Run `ImageCrop` with the four bound columns given as slices — the shape the
    /// dispatcher builds after evaluating each bound expression against the batch.
    fn crop_at(arr: &ArrayRef, x: &[i64], y: &[i64], w: &[i64], h: &[i64]) -> ArrayRef {
        let cols: Vec<ArrayRef> = [x, y, w, h]
            .iter()
            .map(|v| Arc::new(Int64Array::from(v.to_vec())) as ArrayRef)
            .collect();
        let bounds = Bounds::new([&cols[0], &cols[1], &cols[2], &cols[3]]).unwrap();
        eval_image_crop(arr, &bounds).unwrap()
    }

    #[test]
    fn crop_clips_at_the_edge_rather_than_padding() {
        // The deliberate difference from `center_crop`, which zero-pads: `crop` produces
        // an image someone will look at, and inventing black pixels there invents data.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = crop_at(&arr, &[6; 3], &[6; 3], &[100; 3], &[100; 3]);
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        let got = image::load_from_memory(b.value(0)).unwrap();
        assert_eq!(
            (got.width(), got.height()),
            (2, 2),
            "clipped to what exists"
        );
        assert!(b.is_null(1) && b.is_null(2));
    }

    #[test]
    fn a_crop_window_entirely_outside_the_image_is_null() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(8, 8).as_slice())]));
        let out = crop_at(&arr, &[8], &[0], &[4], &[4]);
        assert!(out
            .as_any()
            .downcast_ref::<BinaryArray>()
            .unwrap()
            .is_null(0));
    }

    /// A per-row window is the reason this variant exists: one image, four different
    /// boxes, four different crops. A kernel that read its bounds once for the batch
    /// would return four identical patches and pass every other test in this file.
    #[test]
    fn each_row_gets_its_own_window() {
        use image::{Rgb, RgbImage};

        // A 8x8 image whose four quadrants are four distinct colors.
        let quadrants = [[255u8, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]];
        let mut img = RgbImage::new(8, 8);
        for (x, y, px) in img.enumerate_pixels_mut() {
            let q = usize::from(x >= 4) + 2 * usize::from(y >= 4);
            *px = Rgb(quadrants[q]);
        }
        let src = png_of(img);
        let rows = vec![Some(src.as_slice()); 4];
        let arr: ArrayRef = Arc::new(BinaryArray::from(rows));

        let out = crop_at(&arr, &[0, 4, 0, 4], &[0, 0, 4, 4], &[4; 4], &[4; 4]);
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        for (i, want) in quadrants.iter().enumerate() {
            let got = image::load_from_memory(b.value(i)).unwrap().into_rgb8();
            assert_eq!((got.width(), got.height()), (4, 4), "row {i}");
            assert_eq!(
                &got.get_pixel(0, 0).0,
                want,
                "row {i} took the wrong quadrant"
            );
        }
    }

    /// A window the caller could not supply is a row with no answer, not a failed batch.
    ///
    /// This is the semantic that had to change when the bounds became data. As constants
    /// a negative offset was a *query* error and rightly raised; per row it is one bad
    /// box among thousands — from a detector that declined to predict, or a join that
    /// matched nothing — and failing the batch for it would lose every good row with it.
    #[test]
    fn an_unusable_window_nulls_its_row_and_leaves_the_others() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(8, 8).as_slice()); 5]));
        let out = crop_at(
            &arr,
            &[0, -1, 0, 0, 0],
            &[0, 0, 0, 0, 0],
            &[4, 4, 0, -4, 4],
            &[4, 4, 4, 4, 4],
        );
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0), "a usable window must still produce a crop");
        assert!(b.is_null(1), "a negative offset");
        assert!(b.is_null(2), "a zero extent");
        assert!(b.is_null(3), "a negative extent");
        assert!(b.is_valid(4), "the good row after the bad ones survives");
    }

    /// A null bound nulls its row rather than being read as zero.
    #[test]
    fn a_null_bound_nulls_its_row() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(8, 8).as_slice()); 2]));
        let xs: ArrayRef = Arc::new(Int64Array::from(vec![Some(0), None]));
        let ys: ArrayRef = Arc::new(Int64Array::from(vec![0, 0]));
        let ws: ArrayRef = Arc::new(Int64Array::from(vec![4, 4]));
        let hs: ArrayRef = Arc::new(Int64Array::from(vec![4, 4]));
        let out = eval_image_crop(&arr, &Bounds::new([&xs, &ys, &ws, &hs]).unwrap()).unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0) && b.is_null(1));
    }

    #[test]
    fn convert_reports_the_mode_it_was_asked_for() {
        // Round-trip through `decode`: whatever mode `convert` produces, `decode` must
        // name it, since a caller reads a mode off one and hands it to the other.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(4, 4).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        for (mode, channels) in [("L", 1), ("LA", 2), ("RGB", 3), ("RGBA", 4)] {
            let out = eval_image(
                ImageFunc::Convert,
                &arr,
                ImageArgs {
                    mode: Some(mode),
                    ..args(None, None)
                },
            )
            .unwrap();
            let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
            let (_, _, got_channels, got_mode) = image_header(b.value(0)).unwrap();
            assert_eq!((got_mode, got_channels), (mode, channels), "{mode}");
            assert!(b.is_null(1) && b.is_null(2), "{mode}");
        }
    }

    #[test]
    fn converting_to_grayscale_matches_the_to_grayscale_kernel() {
        // Two paths reach a luma channel; they must not disagree about the weighting.
        use image::{Rgb, RgbImage};
        let mut img = RgbImage::new(2, 2);
        img.put_pixel(0, 0, Rgb([10, 200, 30]));
        img.put_pixel(1, 0, Rgb([255, 0, 0]));
        img.put_pixel(0, 1, Rgb([0, 0, 255]));
        img.put_pixel(1, 1, Rgb([70, 70, 70]));
        let mut src = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut src, image::ImageFormat::Png)
            .unwrap();
        let src = src.into_inner();
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(src.as_slice())]));

        let converted = eval_image(
            ImageFunc::Convert,
            &arr,
            ImageArgs {
                mode: Some("L"),
                ..args(None, None)
            },
        )
        .unwrap();
        let b = converted.as_any().downcast_ref::<BinaryArray>().unwrap();
        let via_convert = image::load_from_memory(b.value(0)).unwrap().into_luma8();

        let grayscaled = ei(ImageFunc::ToGrayscale, &arr, Some(2), Some(2)).unwrap();
        let fsl = grayscaled
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let row = fsl.value(0);
        let via_kernel = row.as_any().downcast_ref::<UInt8Array>().unwrap();
        for i in 0..4 {
            assert_eq!(
                via_convert.as_raw()[i],
                via_kernel.value(i),
                "pixel {i} disagreed"
            );
        }
    }

    #[test]
    fn an_unknown_color_mode_is_an_error() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        assert!(eval_image(
            ImageFunc::Convert,
            &arr,
            ImageArgs {
                mode: Some("CMYK"),
                ..args(None, None)
            }
        )
        .is_err());
    }

    #[test]
    fn encode_rewrites_the_container_and_keeps_the_pixels() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(4, 4).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        for format in ENCODE_FORMATS {
            let out = eval_image(
                ImageFunc::Encode,
                &arr,
                ImageArgs {
                    format: Some(format),
                    ..args(None, None)
                },
            )
            .unwrap();
            let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
            let got = image::load_from_memory(b.value(0)).unwrap();
            assert_eq!((got.width(), got.height()), (4, 4), "{format}");
            // Solid red survives every lossless target; JPEG is lossy, so allow a margin.
            let px = got.into_rgb8();
            let [r, g, bl] = px.get_pixel(0, 0).0;
            assert!(r > 200 && g < 60 && bl < 60, "{format}: {r},{g},{bl}");
            assert!(b.is_null(1) && b.is_null(2), "{format}");
        }
    }

    #[test]
    fn encode_to_jpeg_flattens_an_alpha_channel_rather_than_failing() {
        // JPEG has no alpha. Failing the row would drop every RGBA image in a corpus.
        use image::RgbaImage;
        let mut src = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgba8(RgbaImage::from_pixel(4, 4, image::Rgba([255, 0, 0, 128])))
            .write_to(&mut src, image::ImageFormat::Png)
            .unwrap();
        let src = src.into_inner();
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(src.as_slice())]));
        let out = eval_image(
            ImageFunc::Encode,
            &arr,
            ImageArgs {
                format: Some("jpeg"),
                ..args(None, None)
            },
        )
        .unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0));
        assert_eq!(
            image::load_from_memory(b.value(0))
                .unwrap()
                .color()
                .channel_count(),
            3
        );
    }

    #[test]
    fn an_unknown_encode_format_is_an_error_not_a_null_column() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        assert!(eval_image(
            ImageFunc::Encode,
            &arr,
            ImageArgs {
                format: Some("webp"),
                ..args(None, None)
            }
        )
        .is_err());
    }

    /// A format that cannot carry orientation reports "upright" rather than an error or a
    /// null. `1` is what the Exif code *means* — not "absent", but "no transform needed" —
    /// and null would push every caller into a coalesce for no information.
    ///
    /// The eight transforms themselves are proved against `PIL.ImageOps.exif_transpose` in
    /// `tests/integration/test_image_orientation.py`. They need a JPEG carrying a real Exif
    /// chunk, and the `image` crate reads Exif but does not write it, so the reference
    /// implementation has to supply the fixture.
    #[test]
    fn an_image_without_exif_reports_upright() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(4, 6).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = ei(ImageFunc::ExifOrientation, &arr, None, None).unwrap();
        let codes = out.as_any().downcast_ref::<Int32Array>().unwrap();
        assert_eq!(codes.value(0), 1);
        assert!(codes.is_null(1) && codes.is_null(2));
    }

    /// Orienting an image that needs no orienting must not change it, since a corpus is
    /// oriented wholesale and most of it is already upright.
    #[test]
    fn auto_orient_leaves_an_upright_image_alone() {
        let src = red_png(4, 6);
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(src.as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = ei(ImageFunc::AutoOrient, &arr, None, None).unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        let got = image::load_from_memory(b.value(0)).unwrap();
        assert_eq!((got.width(), got.height()), (4, 6));
        assert_eq!(got.into_rgb8().get_pixel(0, 0).0, [255, 0, 0]);
        assert!(b.is_null(1) && b.is_null(2));
    }

    #[test]
    fn decode_reports_channels_and_mode_from_the_same_header_read() {
        use arrow::array::StringArray;
        use image::{DynamicImage, GrayImage, RgbImage, RgbaImage};

        // One image per color mode, each encoded to PNG (which preserves the mode).
        fn png(img: DynamicImage) -> Vec<u8> {
            let mut buf = Vec::new();
            img.write_to(&mut Cursor::new(&mut buf), image::ImageFormat::Png)
                .unwrap();
            buf
        }
        let gray = png(DynamicImage::ImageLuma8(GrayImage::new(2, 2)));
        let rgb = png(DynamicImage::ImageRgb8(RgbImage::new(2, 2)));
        let rgba = png(DynamicImage::ImageRgba8(RgbaImage::new(2, 2)));
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(gray.as_slice()),
            Some(rgb.as_slice()),
            Some(rgba.as_slice()),
        ]));
        let out = ei(ImageFunc::Decode, &arr, None, None).unwrap();
        let st = out.as_any().downcast_ref::<StructArray>().unwrap();
        let channels = st.column(2).as_any().downcast_ref::<Int32Array>().unwrap();
        let modes = st.column(3).as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(
            (0..3).map(|i| channels.value(i)).collect::<Vec<_>>(),
            vec![1, 3, 4]
        );
        assert_eq!(
            (0..3).map(|i| modes.value(i)).collect::<Vec<_>>(),
            vec!["L", "RGB", "RGBA"]
        );
    }

    #[test]
    fn the_channel_count_always_matches_the_mode_name() {
        use image::ColorType;
        // The two fields are derived together and must not be able to disagree: a caller
        // branching on `mode` and one branching on `channels` have to route the same rows.
        for (color, want) in [
            (ColorType::L8, ("L", 1)),
            (ColorType::L16, ("L", 1)),
            (ColorType::La8, ("LA", 2)),
            (ColorType::Rgb8, ("RGB", 3)),
            (ColorType::Rgb16, ("RGB", 3)),
            (ColorType::Rgba8, ("RGBA", 4)),
            (ColorType::Rgba32F, ("RGBA", 4)),
        ] {
            let (channels, mode) = color_mode(color);
            assert_eq!((mode, channels), want, "{color:?}");
            assert_eq!(
                channels,
                i32::from(color.channel_count()),
                "{color:?}: name and count disagree"
            );
        }
    }

    #[test]
    fn center_crop_takes_the_middle_and_pads_when_smaller() {
        use image::{Rgb, RgbImage};
        // An 8x8 image: left half red, right half green — so a centered 4x4 crop straddles
        // the seam and its columns split red/green, proving we took the middle, not a corner.
        let mut img = RgbImage::new(8, 8);
        for (x, _y, px) in img.enumerate_pixels_mut() {
            *px = if x < 4 {
                Rgb([255, 0, 0])
            } else {
                Rgb([0, 255, 0])
            };
        }
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(png_of(img).as_slice()), None]));
        let out = ei(ImageFunc::CenterCrop, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 4 * 4 * 3);
        assert!(fsl.is_valid(0) && fsl.is_null(1));
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
        // Crop window covers source x in [2,6): first crop column (x=2) is red, last (x=5) green.
        assert_eq!((px.value(0), px.value(1), px.value(2)), (255, 0, 0));
        let last = (4 * 3) - 3; // start of the 4th pixel in row 0
        assert_eq!(
            (px.value(last), px.value(last + 1), px.value(last + 2)),
            (0, 255, 0)
        );

        // A crop larger than the image zero-pads the border (torchvision CenterCrop).
        let small: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(2, 2).as_slice())]));
        let padded = ei(ImageFunc::CenterCrop, &small, Some(4), Some(4)).unwrap();
        let pf = padded
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .unwrap();
        let r = pf.value(0);
        let p = r.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert_eq!(p.value(0), 0); // top-left corner is padding
    }

    #[test]
    fn to_grayscale_reduces_to_one_luma_channel() {
        // A solid red image → luma = round(299*255/1000) = round(76.245) = 76 everywhere.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
        ]));
        let out = ei(ImageFunc::ToGrayscale, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 4 * 4); // one byte per pixel, not *3
        assert!(fsl.is_valid(0) && fsl.is_null(1));
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
        for k in 0..16 {
            assert_eq!(px.value(k), 76, "pixel {k}");
        }
    }

    #[test]
    fn to_tensor_decodes_and_resizes() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
        ]));
        let out = ei(ImageFunc::ToTensor, &arr, Some(4), Some(4)).unwrap();
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
    fn to_tensor_f32_scales_normalizes_and_lays_out() {
        use arrow::array::Float32Array;
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
        ]));

        // Bare: just /255. Solid red → first pixel (1.0, 0.0, 0.0), HWC.
        let out = eval_image(ImageFunc::ToTensorF32, &arr, args(Some(2), Some(2))).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 2 * 2 * 3);
        assert!(fsl.is_valid(0) && fsl.is_null(1));
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<Float32Array>().unwrap();
        assert_eq!(px.value(0), 1.0); // R
        assert_eq!(px.value(1), 0.0); // G
        assert_eq!(px.value(2), 0.0); // B (HWC: first three are the first pixel's RGB)

        // Normalized + channels-first: value = (channel/255 - mean)/std, laid out CHW so
        // the whole first plane is the red channel.
        let mean = [0.5f64, 0.5, 0.5];
        let std = [0.25f64, 0.25, 0.25];
        let out = eval_image(
            ImageFunc::ToTensorF32,
            &arr,
            ImageArgs {
                mean: Some(&mean),
                std: Some(&std),
                channels_first: true,
                ..args(Some(2), Some(2))
            },
        )
        .unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<Float32Array>().unwrap();
        // CHW: indices [0, hw) are the red plane (all 1.0 → (1-0.5)/0.25 = 2.0),
        // [hw, 2hw) green (0 → (0-0.5)/0.25 = -2.0).
        let hw = 2 * 2;
        assert_eq!(px.value(0), 2.0);
        assert_eq!(px.value(hw), -2.0);
        assert_eq!(px.value(2 * hw), -2.0);
    }

    #[test]
    fn to_tensor_f32_rejects_bad_params() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        // Wrong-length mean.
        let bad_mean = [0.5f64, 0.5];
        assert!(eval_image(
            ImageFunc::ToTensorF32,
            &arr,
            ImageArgs {
                mean: Some(&bad_mean),
                ..args(Some(2), Some(2))
            }
        )
        .is_err());
        // Zero std.
        let zero_std = [1.0f64, 0.0, 1.0];
        assert!(eval_image(
            ImageFunc::ToTensorF32,
            &arr,
            ImageArgs {
                std: Some(&zero_std),
                ..args(Some(2), Some(2))
            }
        )
        .is_err());
        // Normalization params on a non-f32 op are rejected, not silently ignored.
        let mean = [0.5f64, 0.5, 0.5];
        assert!(eval_image(
            ImageFunc::ToTensor,
            &arr,
            ImageArgs {
                mean: Some(&mean),
                ..args(Some(2), Some(2))
            }
        )
        .is_err());
    }

    #[test]
    fn to_tensor_parallel_batch_preserves_order_and_nulls() {
        // A batch above `PAR_ROW_THRESHOLD` so the parallel row map runs. Every third
        // row is null and every fifth is corrupt; the rest are solid-color images whose
        // color encodes their index, so we can prove order is preserved across the fan-out.
        let n = 40usize;
        let mut rows: Vec<Option<Vec<u8>>> = Vec::with_capacity(n);
        for i in 0..n {
            if i % 3 == 0 {
                rows.push(None);
            } else if i % 5 == 0 {
                rows.push(Some(b"not an image".to_vec()));
            } else {
                let c = (i % 256) as u8;
                let img = image::RgbImage::from_pixel(6, 6, image::Rgb([c, 0, 0]));
                let mut out = Cursor::new(Vec::new());
                image::DynamicImage::ImageRgb8(img)
                    .write_to(&mut out, image::ImageFormat::Png)
                    .unwrap();
                rows.push(Some(out.into_inner()));
            }
        }
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(rows));
        let out = ei(ImageFunc::ToTensor, &arr, Some(4), Some(4)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.len(), n);
        for i in 0..n {
            if i % 3 == 0 || i % 5 == 0 {
                assert!(fsl.is_null(i), "row {i} should be null");
            } else {
                assert!(fsl.is_valid(i), "row {i} should be valid");
                let row = fsl.value(i);
                let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
                // The solid color survives resize, proving this slot got row `i`'s pixels.
                assert_eq!(px.value(0), (i % 256) as u8, "row {i} decoded out of order");
                assert_eq!(px.value(1), 0);
            }
        }
    }

    #[test]
    fn scaled_decode_downscales_large_jpeg_correctly() {
        // A large solid-color JPEG (source 512×512 ≥ 2× target 128 → the DCT-scaled path
        // runs). The color must survive decode+resize, proving the scaled path is wired
        // correctly and produces the right shape.
        let src = image::RgbImage::from_pixel(512, 512, image::Rgb([40, 160, 200]));
        let mut jpg = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(src)
            .write_to(&mut jpg, image::ImageFormat::Jpeg)
            .unwrap();
        let jpg = jpg.into_inner();

        // The scaled path is taken and returns the right shape.
        let scaled = decode_jpeg_scaled(&jpg, 128, 128);
        assert!(scaled.is_some(), "512→128 JPEG should use the scaled path");
        let (_, sw, sh) = scaled.unwrap();
        assert!(
            sw >= 128 && sh >= 128,
            "scaled output must still cover the target"
        );

        // End to end: the tensor is 128×128×3 and stays the source color (± JPEG noise).
        let arr: ArrayRef = Arc::new(BinaryArray::from_iter(vec![Some(jpg.as_slice())]));
        let out = ei(ImageFunc::ToTensor, &arr, Some(128), Some(128)).unwrap();
        let fsl = out.as_any().downcast_ref::<FixedSizeListArray>().unwrap();
        assert_eq!(fsl.value_length(), 128 * 128 * 3);
        assert!(fsl.is_valid(0));
        let row = fsl.value(0);
        let px = row.as_any().downcast_ref::<UInt8Array>().unwrap();
        assert!(
            (px.value(0) as i32 - 40).abs() < 20,
            "R ~40, got {}",
            px.value(0)
        );
        assert!(
            (px.value(1) as i32 - 160).abs() < 20,
            "G ~160, got {}",
            px.value(1)
        );
        assert!(
            (px.value(2) as i32 - 200).abs() < 20,
            "B ~200, got {}",
            px.value(2)
        );

        // A small JPEG (≤2× target) must NOT take the scaled path — full decode is faster.
        let small = image::RgbImage::from_pixel(130, 130, image::Rgb([10, 20, 30]));
        let mut sbuf = Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(small)
            .write_to(&mut sbuf, image::ImageFormat::Jpeg)
            .unwrap();
        assert!(
            decode_jpeg_scaled(&sbuf.into_inner(), 128, 128).is_none(),
            "130→128 is not a ≥2× downscale; the scaled path should decline it"
        );
    }

    #[test]
    fn out_of_range_dimensions_are_rejected_not_wrapped() {
        // A dimension is `i64` in the IR. A plain `as u32` cast silently wraps:
        //   * a value past u32::MAX → a small one, silently producing a wrong-size tensor;
        //   * a negative value → ~4 billion, an unbounded allocation that aborts the process.
        // Both must be rejected with a clear error rather than wrapped.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        for &bad in &[-1_i64, 0, i64::from(u32::MAX) + 1, i64::MAX] {
            assert!(
                ei(ImageFunc::ToTensor, &arr, Some(bad), Some(4)).is_err(),
                "to_tensor width {bad} should be rejected"
            );
            assert!(
                ei(ImageFunc::ToTensor, &arr, Some(4), Some(bad)).is_err(),
                "to_tensor height {bad} should be rejected"
            );
            assert!(
                ei(ImageFunc::Resize, &arr, Some(bad), Some(4)).is_err(),
                "resize width {bad} should be rejected"
            );
        }
        // The error names the arg but never the (potentially huge) allocation it prevented.
        let err = ei(ImageFunc::ToTensor, &arr, Some(-1), Some(4)).unwrap_err();
        assert!(err.to_string().contains("width"), "{err}");
    }

    #[test]
    fn to_tensor_rejects_dimensions_whose_product_overflows() {
        // Each dimension is a valid `u32`, but `w * h * 3` exceeds `i32::MAX` (the
        // FixedSizeList element length). The old `(w as usize) * (h as usize) * 3` then
        // `per_row as i32` wrapped negative and pre-allocated a multi-GB buffer; it must
        // now be a clean error, returned *before* any allocation.
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![Some(red_png(4, 4).as_slice())]));
        // 40_000 * 40_000 * 3 = 4.8e12 > i32::MAX, but each dim is well within u32.
        let err = ei(ImageFunc::ToTensor, &arr, Some(40_000), Some(40_000)).unwrap_err();
        assert!(err.to_string().contains("exceeds"), "{err}");
        // Just over the boundary: 26_756 * 26_756 * 3 = 2_147_449_008 > i32::MAX (2_147_483_647)?
        // 26_756^2 * 3 = 2_147_608_... let's use a clearly-over value.
        assert!(ei(ImageFunc::ToTensor, &arr, Some(30_000), Some(30_000)).is_err());
        // A legitimate, bounded request still succeeds.
        assert!(ei(ImageFunc::ToTensor, &arr, Some(64), Some(64)).is_ok());
        // `resize` guards the same product (its per-row RGB buffer is `w * h * 3`).
        assert!(ei(ImageFunc::Resize, &arr, Some(40_000), Some(40_000)).is_err());
        assert!(ei(ImageFunc::Resize, &arr, Some(8), Some(8)).is_ok());
    }

    #[test]
    fn resize_reencodes_at_new_size() {
        let arr: ArrayRef = Arc::new(BinaryArray::from(vec![
            Some(red_png(8, 8).as_slice()),
            None,
            Some(b"not an image".as_slice()),
        ]));
        let out = ei(ImageFunc::Resize, &arr, Some(4), Some(2)).unwrap();
        let b = out.as_any().downcast_ref::<BinaryArray>().unwrap();
        assert!(b.is_valid(0));
        // The re-encoded PNG decodes back to the requested 4×2 dimensions.
        let (w, h, _, _) = image_header(b.value(0)).unwrap();
        assert_eq!((w, h), (4, 2));
        assert!(b.is_null(1)); // null input → null
        assert!(b.is_null(2)); // undecodable input → null
    }
}
