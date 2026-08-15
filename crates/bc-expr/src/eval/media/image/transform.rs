//! Geometry and photometric transforms: the augmentation and correction vocabulary.
//!
//! Every op here shares one shape — decode, transform the pixels, re-encode — and one
//! reason to exist: without them a vision pipeline that needs a horizontal flip, a
//! brightness jitter, or an exposure fix has to leave the engine and loop over rows in
//! Python with Pillow, which is precisely the hot-path tuple touch the architecture
//! forbids. They are named and parameterized after `PIL.ImageEnhance` / `PIL.ImageOps`
//! because that is the vocabulary an augmentation policy is already written in, so a
//! torchvision or AutoAugment recipe ports over without rethinking what a factor means.
//!
//! The photometric ops work on RGBA8 and restore the source's channel count afterwards.
//! Doing it the other way round — flattening to RGB up front — would silently drop the
//! alpha channel of a PNG corpus, which is the kind of loss that shows up much later as a
//! black halo around every cut-out object.

use arrow::array::{Array, ArrayRef, GenericBinaryArray, OffsetSizeTrait};

use super::reencode::{assemble_binary, Output};
use super::{map_rows, rec601, ImageArgs};
use crate::{ExprError, ImageFunc};

/// The scalar knob an op takes, validated once for the batch.
///
/// Validating here rather than per row is the same decision `Output::resolve` makes, for
/// the same reason: a factor out of range is the caller's mistake, and reporting it once
/// per row of a million-row corpus buries it.
fn factor(func: ImageFunc, value: Option<f64>, default: f64) -> Result<f64, ExprError> {
    let v = value.unwrap_or(default);
    if !v.is_finite() {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("factor must be a finite number, got {v}"),
        });
    }
    Ok(v)
}

/// Evaluate one geometry/photometric op over a column of encoded image bytes.
pub(super) fn eval<O: OffsetSizeTrait>(
    func: ImageFunc,
    bytes: &GenericBinaryArray<O>,
    arg: Option<f64>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let op = Op::resolve(func, arg)?;
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let img = image::load_from_memory(bytes.value(i)).ok()?;
        out.write(op.apply(img))
    });
    Ok(assemble_binary(rows))
}

/// A resolved transform: the op plus its already-validated parameter.
#[derive(Debug, Clone, Copy)]
enum Op {
    Rotate(Quarter),
    FlipHorizontal,
    FlipVertical,
    Brightness(f32),
    Contrast(f32),
    Saturation(f32),
    Hue(i32),
    Blur(f32),
    Sharpen(f32),
    Invert,
    Posterize(u8),
    Solarize(u8),
    Equalize,
    AutoContrast(f64),
}

/// A right-angle rotation. Only right angles: a free rotation resamples every pixel and
/// leaves a triangular border in a colour nobody chose, where 90/180/270 is a
/// transposition that is exact, lossless, and what "rotate this corpus upright" means.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Quarter {
    None,
    Cw90,
    Cw180,
    Cw270,
}

impl Op {
    fn resolve(func: ImageFunc, arg: Option<f64>) -> Result<Self, ExprError> {
        Ok(match func {
            ImageFunc::Rotate => Self::Rotate(Quarter::resolve(factor(func, arg, 0.0)?)?),
            ImageFunc::FlipHorizontal => Self::FlipHorizontal,
            ImageFunc::FlipVertical => Self::FlipVertical,
            ImageFunc::AdjustBrightness => Self::Brightness(non_negative(func, arg, 1.0)? as f32),
            ImageFunc::AdjustContrast => Self::Contrast(non_negative(func, arg, 1.0)? as f32),
            ImageFunc::AdjustSaturation => Self::Saturation(non_negative(func, arg, 1.0)? as f32),
            // Degrees wrap, so any finite value is meaningful; `huerotate` takes an i32.
            ImageFunc::AdjustHue => Self::Hue(factor(func, arg, 0.0)?.rem_euclid(360.0) as i32),
            ImageFunc::Blur => Self::Blur(non_negative(func, arg, 1.0)? as f32),
            ImageFunc::Sharpen => Self::Sharpen(non_negative(func, arg, 1.0)? as f32),
            ImageFunc::Invert => Self::Invert,
            ImageFunc::Posterize => Self::Posterize(bounded(func, arg, 4.0, 1.0, 8.0)? as u8),
            ImageFunc::Solarize => Self::Solarize(bounded(func, arg, 128.0, 0.0, 255.0)? as u8),
            ImageFunc::Equalize => Self::Equalize,
            ImageFunc::AutoContrast => Self::AutoContrast(bounded(func, arg, 0.0, 0.0, 49.0)?),
            other => {
                return Err(ExprError::InvalidArgument {
                    func: format!("{other:?}"),
                    reason: "not a geometry or photometric transform".to_string(),
                })
            }
        })
    }

    fn apply(self, img: image::DynamicImage) -> image::DynamicImage {
        use image::imageops;
        // `imageops`' geometry helpers are generic over the view and hand back RGBA
        // whatever went in, so a flipped RGB corpus would silently grow a fourth channel
        // — a third more bytes per row, a changed column type, and an alpha plane the
        // caller never had. Restoring the source's channel count is what keeps a geometry
        // op a geometry op.
        let had_alpha = img.color().has_alpha();
        let moved = match self {
            Self::Rotate(Quarter::None) => return img,
            Self::Rotate(Quarter::Cw90) => imageops::rotate90(&img),
            Self::Rotate(Quarter::Cw180) => imageops::rotate180(&img),
            Self::Rotate(Quarter::Cw270) => imageops::rotate270(&img),
            Self::FlipHorizontal => imageops::flip_horizontal(&img),
            Self::FlipVertical => imageops::flip_vertical(&img),
            _ => return self.photometric(img),
        };
        restore_channels(moved, had_alpha)
    }

    /// The pixel-value ops, applied over RGBA8 and restored to the source's channel count.
    fn photometric(self, img: image::DynamicImage) -> image::DynamicImage {
        let had_alpha = img.color().has_alpha();
        let mut rgba = match self {
            // `huerotate` is the one op the `image` crate already states exactly as PIL
            // does, so it is used rather than restated.
            Self::Hue(deg) => image::imageops::huerotate(&img, deg),
            // A Gaussian blur is a separable convolution over the whole plane; `fast_blur`
            // is the three-box-pass approximation of it, which is what makes it usable on a
            // corpus rather than on one image. Deterministic, so the parallel path and the
            // oracle cannot disagree.
            Self::Blur(sigma) if sigma > 0.0 => image::imageops::fast_blur(&img.to_rgba8(), sigma),
            _ => img.to_rgba8(),
        };
        match self {
            Self::Brightness(f) => scale_channels(&mut rgba, |v| v * f),
            Self::Contrast(f) => {
                let mean = mean_luma(&rgba);
                scale_channels(&mut rgba, |v| mean + (v - mean) * f);
            }
            Self::Saturation(f) => saturate(&mut rgba, f),
            Self::Sharpen(amount) if amount > 0.0 => unsharp(&mut rgba, amount),
            Self::Invert => {
                for px in rgba.pixels_mut() {
                    for c in 0..3 {
                        px.0[c] = 255 - px.0[c];
                    }
                }
            }
            Self::Posterize(bits) => {
                // Keep the top `bits` bits and clear the rest. Masking rather than
                // quantizing-and-rescaling is what PIL does, so 1 bit gives 0 and 128.
                let mask = !0u8 << (8 - bits);
                for px in rgba.pixels_mut() {
                    for c in 0..3 {
                        px.0[c] &= mask;
                    }
                }
            }
            Self::Solarize(threshold) => {
                for px in rgba.pixels_mut() {
                    for c in 0..3 {
                        if px.0[c] >= threshold {
                            px.0[c] = 255 - px.0[c];
                        }
                    }
                }
            }
            Self::Equalize => equalize(&mut rgba),
            Self::AutoContrast(cutoff) => autocontrast(&mut rgba, cutoff),
            _ => {}
        }
        restore_channels(rgba, had_alpha)
    }
}

/// Put an RGBA working buffer back into the channel count the source had.
///
/// Every op here works in RGBA so an alpha channel survives the arithmetic, but handing
/// that back for an RGB source would add a plane the caller never had — a third more bytes
/// in every row and a changed column type, from an operation that was supposed to move or
/// rescale pixels.
fn restore_channels(rgba: image::RgbaImage, had_alpha: bool) -> image::DynamicImage {
    if had_alpha {
        image::DynamicImage::ImageRgba8(rgba)
    } else {
        image::DynamicImage::ImageRgb8(image::DynamicImage::ImageRgba8(rgba).into_rgb8())
    }
}

impl Quarter {
    fn resolve(degrees: f64) -> Result<Self, ExprError> {
        // Normalize first, so `-90` and `630` are ordinary inputs rather than errors.
        let d = degrees.rem_euclid(360.0);
        Ok(match d {
            _ if d == 0.0 => Self::None,
            _ if d == 90.0 => Self::Cw90,
            _ if d == 180.0 => Self::Cw180,
            _ if d == 270.0 => Self::Cw270,
            _ => {
                return Err(ExprError::InvalidArgument {
                    func: "Rotate".to_string(),
                    reason: format!(
                        "degrees must be a multiple of 90 (a free rotation would resample \
                         every pixel and pad the corners), got {degrees}"
                    ),
                })
            }
        })
    }
}

fn non_negative(func: ImageFunc, value: Option<f64>, default: f64) -> Result<f64, ExprError> {
    let v = factor(func, value, default)?;
    if v < 0.0 {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("factor must be >= 0, got {v}"),
        });
    }
    Ok(v)
}

fn bounded(
    func: ImageFunc,
    value: Option<f64>,
    default: f64,
    lo: f64,
    hi: f64,
) -> Result<f64, ExprError> {
    let v = factor(func, value, default)?;
    if !(lo..=hi).contains(&v) {
        return Err(ExprError::InvalidArgument {
            func: format!("{func:?}"),
            reason: format!("value must be in {lo}..={hi}, got {v}"),
        });
    }
    Ok(v)
}

/// Apply `f` to each colour channel in 0..=255 float space, clamped back to a byte.
fn scale_channels(img: &mut image::RgbaImage, f: impl Fn(f32) -> f32) {
    for px in img.pixels_mut() {
        for c in 0..3 {
            px.0[c] = f(px.0[c] as f32).clamp(0.0, 255.0).round() as u8;
        }
    }
}

/// The image's mean Rec.601 luma — the grey `adjust_contrast` pivots around, matching
/// `PIL.ImageEnhance.Contrast`, which builds its degenerate image from the mean of the
/// grayscale conversion.
fn mean_luma(img: &image::RgbaImage) -> f32 {
    if img.width() == 0 || img.height() == 0 {
        return 0.0;
    }
    let total: u64 = img
        .pixels()
        .map(|p| u64::from(rec601([p.0[0], p.0[1], p.0[2]])))
        .sum();
    total as f32 / (img.width() as f32 * img.height() as f32)
}

/// Interpolate each pixel between its grey and its colour: `0` grayscale, `1` identity.
fn saturate(img: &mut image::RgbaImage, f: f32) {
    for px in img.pixels_mut() {
        let grey = rec601([px.0[0], px.0[1], px.0[2]]) as f32;
        for c in 0..3 {
            px.0[c] = (grey + (px.0[c] as f32 - grey) * f)
                .clamp(0.0, 255.0)
                .round() as u8;
        }
    }
}

/// An unsharp mask: `out = in + amount * (in - blur(in))`.
///
/// Stated as the classical formula rather than reached for from `imageops::unsharpen`,
/// whose `threshold` parameter is an integer difference cutoff with no counterpart in the
/// `amount` an augmentation policy specifies.
fn unsharp(img: &mut image::RgbaImage, amount: f32) {
    let blurred = image::imageops::fast_blur(img, 1.0);
    for (px, soft) in img.pixels_mut().zip(blurred.pixels()) {
        for c in 0..3 {
            let sharp = px.0[c] as f32 + amount * (px.0[c] as f32 - soft.0[c] as f32);
            px.0[c] = sharp.clamp(0.0, 255.0).round() as u8;
        }
    }
}

/// Per-channel histogram of the colour planes (alpha is never touched).
fn histograms(img: &image::RgbaImage) -> [[u32; 256]; 3] {
    let mut hist = [[0u32; 256]; 3];
    for px in img.pixels() {
        for (c, h) in hist.iter_mut().enumerate() {
            h[px.0[c] as usize] += 1;
        }
    }
    hist
}

/// Per-channel histogram equalization, matching `PIL.ImageOps.equalize`.
fn equalize(img: &mut image::RgbaImage) {
    let hist = histograms(img);
    let mut luts = [[0u8; 256]; 3];
    for (c, lut) in luts.iter_mut().enumerate() {
        let total: u32 = hist[c].iter().sum();
        if total == 0 {
            continue;
        }
        // PIL's step: the average bin population excluding the last non-empty bin. A
        // channel flatter than one step is left alone rather than being stretched into
        // noise, which is what keeps a solid-colour tile from equalizing to garbage.
        let last = hist[c].iter().rposition(|&n| n > 0).unwrap_or(0);
        let step = (total - hist[c][last]) / 255;
        if step == 0 {
            for (v, out) in lut.iter_mut().enumerate() {
                *out = v as u8;
            }
            continue;
        }
        let mut acc = step / 2;
        for (v, out) in lut.iter_mut().enumerate() {
            *out = (acc / step).min(255) as u8;
            acc += hist[c][v];
        }
    }
    for px in img.pixels_mut() {
        for c in 0..3 {
            px.0[c] = luts[c][px.0[c] as usize];
        }
    }
}

/// Linear per-channel range stretch, ignoring `cutoff` percent of each tail
/// (`PIL.ImageOps.autocontrast`).
fn autocontrast(img: &mut image::RgbaImage, cutoff: f64) {
    let hist = histograms(img);
    let mut luts = [[0u8; 256]; 3];
    for (c, lut) in luts.iter_mut().enumerate() {
        let total: u64 = hist[c].iter().map(|&n| u64::from(n)).sum();
        let drop = ((total as f64) * cutoff / 100.0) as u64;
        let lo = trim(&hist[c], drop, false);
        let hi = trim(&hist[c], drop, true);
        // A channel whose surviving range is a single value has nothing to stretch;
        // scaling it would divide by zero and turn a flat plane into an arbitrary one.
        if hi <= lo {
            for (v, out) in lut.iter_mut().enumerate() {
                *out = v as u8;
            }
            continue;
        }
        let scale = 255.0 / f64::from(hi - lo);
        for (v, out) in lut.iter_mut().enumerate() {
            *out = (((v as f64) - f64::from(lo)) * scale)
                .clamp(0.0, 255.0)
                .round() as u8;
        }
    }
    for px in img.pixels_mut() {
        for c in 0..3 {
            px.0[c] = luts[c][px.0[c] as usize];
        }
    }
}

/// The first (or last) histogram value left after discarding `drop` samples from that end.
fn trim(hist: &[u32; 256], drop: u64, from_top: bool) -> u32 {
    let mut left = drop;
    let order: Box<dyn Iterator<Item = usize>> = if from_top {
        Box::new((0..256).rev())
    } else {
        Box::new(0..256)
    };
    let mut last = if from_top { 255 } else { 0 };
    for v in order {
        last = v;
        let n = u64::from(hist[v]);
        if n > left {
            return v as u32;
        }
        left -= n;
    }
    last as u32
}

/// `pad(width, height, fill)` → the image centered on a filled canvas, **unscaled**.
///
/// The difference from `letterbox` is that nothing is resampled. A corpus of unequal-size
/// crops becomes batchable at its true pixel values, which is what a super-resolution or
/// OCR pipeline needs and what a scaling pad quietly destroys. A canvas smaller than the
/// image crops it centrally, so the op is total rather than failing on the one row that
/// happens to be larger than the target.
pub(super) fn pad<O: OffsetSizeTrait>(
    bytes: &GenericBinaryArray<O>,
    args: ImageArgs<'_>,
    out: Output,
) -> Result<ArrayRef, ExprError> {
    let w = super::dim("pad", "width", args.width)?;
    let h = super::dim("pad", "height", args.height)?;
    super::element_len_guard("pad", u64::from(w) * u64::from(h) * 3, "bytes")?;
    let fill = match args.fill {
        None => 0u8,
        Some(v) => u8::try_from(v).map_err(|_| ExprError::InvalidArgument {
            func: "pad".to_string(),
            reason: format!("fill must be a byte value in 0..=255, got {v}"),
        })?,
    };
    let rows: Vec<Option<Vec<u8>>> = map_rows(bytes.len(), |i| {
        if bytes.is_null(i) {
            return None;
        }
        let src = image::load_from_memory(bytes.value(i)).ok()?.into_rgb8();
        let (sw, sh) = src.dimensions();
        let mut canvas = image::RgbImage::from_pixel(w, h, image::Rgb([fill, fill, fill]));
        // Signed offsets: a source larger than the canvas has a negative one, which is
        // the crop case, and the intersection loop below covers both without a branch.
        let dx = (i64::from(w) - i64::from(sw)) / 2;
        let dy = (i64::from(h) - i64::from(sh)) / 2;
        for y in 0..i64::from(sh) {
            let ty = y + dy;
            if ty < 0 || ty >= i64::from(h) {
                continue;
            }
            for x in 0..i64::from(sw) {
                let tx = x + dx;
                if tx < 0 || tx >= i64::from(w) {
                    continue;
                }
                canvas.put_pixel(tx as u32, ty as u32, *src.get_pixel(x as u32, y as u32));
            }
        }
        out.write(image::DynamicImage::ImageRgb8(canvas))
    });
    Ok(assemble_binary(rows))
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{AsArray, BinaryArray};

    fn png(w: u32, h: u32, f: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        let img = image::RgbImage::from_fn(w, h, |x, y| image::Rgb(f(x, y)));
        let mut out = std::io::Cursor::new(Vec::new());
        image::DynamicImage::ImageRgb8(img)
            .write_to(&mut out, image::ImageFormat::Png)
            .unwrap();
        out.into_inner()
    }

    fn run(func: ImageFunc, arg: Option<f64>, img: &[u8]) -> image::RgbImage {
        let arr = BinaryArray::from_iter(vec![Some(img)]);
        let out = eval(func, &arr, arg, Output::resolve(func, None, None).unwrap()).unwrap();
        let b = out.as_binary::<i32>();
        image::load_from_memory(b.value(0)).unwrap().into_rgb8()
    }

    /// The corner that pins orientation: a rotation must move a known pixel to a known
    /// place, not merely produce an image of the transposed size.
    #[test]
    fn rotate_turns_the_image_clockwise() {
        // Top-left is white, everything else black.
        let src = png(4, 2, |x, y| {
            if x == 0 && y == 0 {
                [255, 255, 255]
            } else {
                [0, 0, 0]
            }
        });
        let out = run(ImageFunc::Rotate, Some(90.0), &src);
        assert_eq!(out.dimensions(), (2, 4));
        // A clockwise quarter turn sends (0, 0) to the top-*right*.
        assert_eq!(out.get_pixel(1, 0).0, [255, 255, 255]);
    }

    /// Negative and over-full-turn angles are ordinary inputs, not errors.
    #[test]
    fn rotation_angles_are_normalized() {
        let src = png(4, 2, |x, y| [(x * 60) as u8, (y * 60) as u8, 0]);
        let a = run(ImageFunc::Rotate, Some(-90.0), &src);
        let b = run(ImageFunc::Rotate, Some(270.0), &src);
        assert_eq!(a.as_raw(), b.as_raw());
        let identity = image::load_from_memory(&src).unwrap().into_rgb8();
        assert_eq!(
            run(ImageFunc::Rotate, Some(360.0), &src).as_raw(),
            identity.as_raw()
        );
    }

    #[test]
    fn a_free_rotation_is_refused_rather_than_resampled() {
        let arr = BinaryArray::from_iter(vec![Some(png(2, 2, |_, _| [1, 2, 3]).as_slice())]);
        let out = Output::resolve(ImageFunc::Rotate, None, None).unwrap();
        let err = eval(ImageFunc::Rotate, &arr, Some(45.0), out).unwrap_err();
        assert!(format!("{err}").contains("multiple of 90"), "{err}");
    }

    #[test]
    fn flips_mirror_the_axis_they_name() {
        let src = png(2, 2, |x, y| [(x * 100) as u8, (y * 100) as u8, 0]);
        let h = run(ImageFunc::FlipHorizontal, None, &src);
        assert_eq!(h.get_pixel(0, 0).0[0], 100);
        let v = run(ImageFunc::FlipVertical, None, &src);
        assert_eq!(v.get_pixel(0, 0).0[1], 100);
    }

    /// The `PIL.ImageEnhance` convention every augmentation policy is written against:
    /// 1.0 is the identity, 0.0 the degenerate, and the scale is multiplicative.
    #[test]
    fn a_factor_of_one_is_the_identity() {
        let src = png(8, 8, |x, y| [(x * 30) as u8, (y * 30) as u8, 90]);
        let original = image::load_from_memory(&src).unwrap().into_rgb8();
        for func in [
            ImageFunc::AdjustBrightness,
            ImageFunc::AdjustContrast,
            ImageFunc::AdjustSaturation,
        ] {
            let out = run(func, Some(1.0), &src);
            assert_eq!(out.as_raw(), original.as_raw(), "{func:?} moved a pixel");
        }
    }

    #[test]
    fn brightness_scales_and_saturates_at_white() {
        let src = png(4, 4, |_, _| [100, 100, 100]);
        let at = |f: f64| {
            run(ImageFunc::AdjustBrightness, Some(f), &src)
                .get_pixel(0, 0)
                .0
        };
        assert_eq!(at(0.5), [50, 50, 50]);
        assert_eq!(at(0.0), [0, 0, 0]);
        // Clamped, not wrapped: 100 * 4 is 400, which must land on 255 rather than 144.
        assert_eq!(at(4.0), [255, 255, 255]);
    }

    #[test]
    fn zero_saturation_is_grayscale_and_zero_contrast_is_flat() {
        let src = png(4, 4, |x, _| [(x * 60) as u8, 20, 200]);
        let grey = run(ImageFunc::AdjustSaturation, Some(0.0), &src);
        for px in grey.pixels() {
            assert_eq!(px.0[0], px.0[1]);
            assert_eq!(px.0[1], px.0[2]);
        }
        let flat = run(ImageFunc::AdjustContrast, Some(0.0), &src);
        let first = flat.get_pixel(0, 0).0;
        assert!(
            flat.pixels().all(|p| p.0 == first),
            "contrast 0 left variation"
        );
    }

    #[test]
    fn invert_is_the_photographic_negative() {
        let src = png(4, 4, |x, y| [(x * 50) as u8, (y * 50) as u8, 77]);
        assert_eq!(
            run(ImageFunc::Invert, None, &src).get_pixel(0, 0).0,
            [255, 255, 178]
        );
    }

    /// Posterize keeps the top bits — 1 bit leaves only 0 and 128, which is the PIL
    /// behaviour a masking implementation gets right and a rescaling one does not.
    #[test]
    fn posterize_masks_the_low_bits() {
        let src = png(4, 1, |x, _| [(x * 64) as u8, 255, 130]);
        let out = run(ImageFunc::Posterize, Some(1.0), &src);
        for px in out.pixels() {
            for c in 0..3 {
                assert!(px.0[c] == 0 || px.0[c] == 128, "got {}", px.0[c]);
            }
        }
        // 8 bits is the identity.
        assert_eq!(
            run(ImageFunc::Posterize, Some(8.0), &src).get_pixel(1, 0).0,
            [64, 255, 130]
        );
    }

    #[test]
    fn solarize_inverts_only_above_the_threshold() {
        let src = png(2, 1, |x, _| {
            if x == 0 {
                [10, 10, 10]
            } else {
                [200, 200, 200]
            }
        });
        let out = run(ImageFunc::Solarize, Some(128.0), &src);
        assert_eq!(out.get_pixel(0, 0).0, [10, 10, 10]);
        assert_eq!(out.get_pixel(1, 0).0, [55, 55, 55]);
    }

    /// The failure mode both histogram ops have: a flat image has no range, and a naive
    /// implementation divides by zero and returns an arbitrary plane.
    #[test]
    fn a_flat_image_survives_equalize_and_autocontrast() {
        let src = png(8, 8, |_, _| [128, 128, 128]);
        for func in [ImageFunc::Equalize, ImageFunc::AutoContrast] {
            let out = run(func, None, &src);
            let first = out.get_pixel(0, 0).0;
            assert!(
                out.pixels().all(|p| p.0 == first),
                "{func:?} invented variation"
            );
        }
    }

    #[test]
    fn autocontrast_stretches_a_narrow_range_to_the_full_one() {
        let src = png(4, 1, |x, _| [(100 + x * 10) as u8; 3]);
        let out = run(ImageFunc::AutoContrast, Some(0.0), &src);
        assert_eq!(out.get_pixel(0, 0).0[0], 0);
        assert_eq!(out.get_pixel(3, 0).0[0], 255);
    }

    #[test]
    fn blur_reduces_variance_and_sharpen_raises_it() {
        let checker = png(32, 32, |x, y| {
            let v = if (x / 4 + y / 4) % 2 == 0 { 0 } else { 255 };
            [v, v, v]
        });
        let variance = |img: &image::RgbImage| {
            let n = img.pixels().count() as f64;
            let mean = img.pixels().map(|p| f64::from(p.0[0])).sum::<f64>() / n;
            img.pixels()
                .map(|p| (f64::from(p.0[0]) - mean).powi(2))
                .sum::<f64>()
                / n
        };
        let base = variance(&image::load_from_memory(&checker).unwrap().into_rgb8());
        assert!(variance(&run(ImageFunc::Blur, Some(3.0), &checker)) < base);
        assert!(variance(&run(ImageFunc::Sharpen, Some(2.0), &checker)) >= base);
    }

    /// The reason `format` was lifted off `encode`: every bytes-out op must be able to
    /// answer in the container the caller wants, in the decode it was already doing.
    #[test]
    fn a_transform_can_emit_any_supported_container() {
        let src = png(8, 8, |x, _| [(x * 30) as u8, 10, 10]);
        let col = BinaryArray::from_iter(vec![Some(src.as_slice())]);
        for (name, want) in [
            ("jpeg", image::ImageFormat::Jpeg),
            ("bmp", image::ImageFormat::Bmp),
        ] {
            let out = Output::resolve(ImageFunc::FlipHorizontal, Some(name), None).unwrap();
            let arr = eval(ImageFunc::FlipHorizontal, &col, None, out).unwrap();
            let bytes = arr.as_binary::<i32>().value(0).to_vec();
            let got = image::ImageReader::new(std::io::Cursor::new(&bytes))
                .with_guessed_format()
                .unwrap()
                .format();
            assert_eq!(got, Some(want));
        }
    }

    /// The point of a quality knob: a lower one must actually produce fewer bytes.
    #[test]
    fn jpeg_quality_trades_size_for_fidelity() {
        let src = png(64, 64, |x, y| {
            [(x * 4) as u8, (y * 4) as u8, ((x ^ y) * 3) as u8]
        });
        let col = BinaryArray::from_iter(vec![Some(src.as_slice())]);
        let size = |q: i64| {
            let out = Output::resolve(ImageFunc::Invert, Some("jpeg"), Some(q)).unwrap();
            eval(ImageFunc::Invert, &col, None, out)
                .unwrap()
                .as_binary::<i32>()
                .value(0)
                .len()
        };
        assert!(
            size(20) < size(95),
            "quality knob did not change the encoding"
        );
    }

    #[test]
    fn pad_centers_without_scaling_and_fills_the_rest() {
        let src = png(2, 2, |_, _| [200, 100, 50]);
        let arr = BinaryArray::from_iter(vec![Some(src.as_slice())]);
        let args = ImageArgs {
            width: Some(4),
            height: Some(4),
            mean: None,
            std: None,
            channels_first: false,
            format: None,
            mode: None,
            quality: None,
            factor: None,
            fill: Some(7),
        };
        let out = pad(
            &arr,
            args,
            Output::resolve(ImageFunc::Pad, None, None).unwrap(),
        )
        .unwrap();
        let img = image::load_from_memory(out.as_binary::<i32>().value(0))
            .unwrap()
            .into_rgb8();
        assert_eq!(img.dimensions(), (4, 4));
        assert_eq!(img.get_pixel(0, 0).0, [7, 7, 7], "corner should be fill");
        // Unscaled: the source pixel value survives exactly rather than being resampled.
        assert_eq!(img.get_pixel(1, 1).0, [200, 100, 50]);
    }

    #[test]
    fn a_null_or_undecodable_row_is_null_rather_than_an_error() {
        let arr = BinaryArray::from_iter(vec![None, Some(b"not an image".as_slice())]);
        let out = Output::resolve(ImageFunc::Invert, None, None).unwrap();
        assert_eq!(
            eval(ImageFunc::Invert, &arr, None, out)
                .unwrap()
                .null_count(),
            2
        );
    }
}
