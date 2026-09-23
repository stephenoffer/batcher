# Transforming and fingerprinting images

This page covers the image operations that change pixels rather than shape them into a tensor: the geometry and color transforms an augmentation policy is written from, and the perceptual hashes a deduplication pass is built on. All of them run in the Rust data plane over a binary column, so a corpus never leaves the engine to be flipped or recolored. Getting the bytes in the first place is covered in {doc}`/ml/preparing/multimodal/decoding`.

## Which output format an operation writes

Every operation that hands back an image takes a `format` and a `quality`, and the choice matters more than it looks. A photographic corpus arrives as JPEG. Re-encoding it as PNG is slower to write and several times larger, so a resize meant to shrink a dataset inflates it instead:

```python
import base64
import batcher as bt

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGP4z8DAAMIM/4EAAB/uBfsL"
    "2WiLAAAAAElFTkSuQmCC"
)
photos = bt.from_pydict({"bytes": [png]})

small = bt.col("bytes").image.resize(64, 64, format="jpeg", quality=80)
print(photos.select(f=small.image.format()).to_pydict())
```

`png` is the default because it's lossless. `quality` applies only to the lossy containers, since the lossless ones have nothing to trade.

## Geometry

`rotate(degrees)` turns the image by a multiple of 90. Only right angles are allowed. A free rotation resamples every pixel and leaves a triangular border in a color nobody chose, while a quarter turn is an exact transposition. Negative and over-full-turn values are normalized, so `-90` and `270` are the same rotation.

`flip_horizontal()` and `flip_vertical()` mirror an axis. The horizontal flip is the most common training-time augmentation, and it belongs on the image column rather than in a loader, so a detector's boxes can be flipped in the same pass.

`pad(width, height, fill=0)` centers the image on a canvas without scaling it. Unlike `letterbox`, which fits and resamples, nothing here is resampled, so every surviving pixel keeps its exact value. OCR and super-resolution pipelines need that. A canvas smaller than the image crops it centrally instead of failing the row.

```python
# docs: skip
from batcher import col

augmented = photos.with_columns(
    flipped=col("bytes").image.flip_horizontal(format="jpeg"),
    upright=col("bytes").image.rotate(-90),
    canvas=col("bytes").image.pad(640, 640, fill=114),
)
```

Flips and rotations keep the channel count, so a flipped RGB image stays RGB. That sounds obvious, and it's what a naive implementation gets wrong: the underlying helpers hand back RGBA whatever went in, so a flipped corpus silently grows a fourth channel and a third more bytes per row. `pad` is the exception, covered under the limitations below.

## Color and tone

Four adjustments follow the `PIL.ImageEnhance` convention, so an augmentation policy written against torchvision ports over unchanged. `1.0` is the identity and `0.0` is the degenerate case, as the following table shows:

| Method | `0.0` gives | What it varies |
|---|---|---|
| `adjust_brightness(factor)` | black | every channel scaled |
| `adjust_contrast(factor)` | a flat field at the image's own mean luma | spread about the mean |
| `adjust_saturation(factor)` | grayscale, still three channels | distance from gray |
| `adjust_hue(degrees)` | no change (degrees wrap) | hue only, at constant saturation and value |

`blur(sigma)` and `sharpen(amount)` move detail rather than color. `blur` is a Gaussian of `sigma` pixels; `sharpen` is the classical unsharp mask, `image + amount * (image - blur(image))`.

Five more come from the AutoAugment and RandAugment policies. `posterize(bits)` keeps the top `bits` bits of each channel by masking, so `bits=1` leaves only 0 and 128. `solarize(threshold)` inverts every channel value at or above the threshold, and `invert()` is the photographic negative. `equalize()` flattens each channel's histogram. `autocontrast(cutoff)` is gentler: it stretches the range without redistributing values within it, ignoring `cutoff` percent of each tail.

```python
# docs: skip
from batcher import col

jittered = photos.select(
    warm=col("bytes").image.adjust_hue(15),
    vivid=col("bytes").image.adjust_saturation(1.4),
    fixed=col("bytes").image.autocontrast(cutoff=2.0),
)
```

`equalize()` and `autocontrast()` both leave a flat image alone instead of dividing by an empty range. A solid-color tile comes out unchanged, not as noise.

## Perceptual hashes and near-duplicate detection

A scraped corpus holds the same picture at three resolutions, in two codecs and under one watermark. A content hash can't find those copies because every byte differs, and a model costs an embedding per image. A perceptual hash costs a small decode and turns the question into an integer comparison:

```python
# docs: skip
from batcher import col

fingerprinted = photos.with_columns(quick=col("bytes").image.ahash())
pairs = fingerprinted.join(fingerprinted.rename({"quick": "other"}), how="cross")
near_duplicates = pairs.filter(col("quick").bitwise_xor(col("other")).bit_count() <= 6)
```

The three hashes trade cost against robustness:

| Method | How it reduces the image | Use it for |
|---|---|---|
| `ahash()` | 8x8 luma, thresholded at its own mean | the cheapest pre-filter |
| `dhash()` | 8x8 comparisons of horizontally adjacent pixels | a middle ground, robust to brightness shifts |
| `phash()` | the 8x8 lowest-frequency DCT coefficients of a 32x32 luma reduction, thresholded at their median | confirming a candidate, the most robust of the three to rescaling, re-encoding and moderate cropping |

All three return an `Int64` whose bits are the hash. `a.bitwise_xor(b).bit_count()` is the Hamming distance, and a threshold on it is a similarity predicate. Block on `ahash`, then confirm with `phash`.

## Reading facts without decoding pixels

A few operations answer from the file header alone, without decoding pixels:

```python
import base64
import batcher as bt

png = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAIAAAD91JpzAAAAEklEQVR4nGP4z8DAAMIM/4EAAB/uBfsL"
    "2WiLAAAAAElFTkSuQmCC"
)
corpus = bt.from_pydict({"bytes": [png]})

print(
    corpus.select(
        ratio=bt.col("bytes").image.aspect_ratio(),
        alpha=bt.col("bytes").image.has_alpha(),
        container=bt.col("bytes").image.format(),
    ).to_pydict()
)
```

`format()` sniffs the magic bytes, not the file extension. That's how you find the `.jpg` files that are really PNGs, which decode fine and then break whatever downstream step branched on the name. It reports the container name, `jpeg` rather than `jpg`. That's the spelling `encode()` accepts and a listing's `format` column reports, so a value read from one works in the others.

`aspect_ratio()` reports null, not infinity, for a zero-height image, so a filter written to find panoramas can't silently accept it. `has_alpha()` tells you whether a corpus needs flattening with `convert("RGB")` before a three-channel model.

## Requirements and limitations

- `rotate` accepts multiples of 90 only. A free rotation is refused at plan build rather than resampled.
- WebP is readable but not writable, so it is not offered as an output `format`.
- `pad` flattens to RGB, because the canvas fill is a single byte value applied to all three channels.
- The hashes are stable across runs and machines: the luma reduction uses integer weights, so a stored hash stays comparable.

## See also

- {doc}`/ml/preparing/multimodal/curating`: the measures that decide which rows are worth keeping, and `dhash` deduplication.
- {doc}`/ml/preparing/preprocessors/deduplication`: deduplicating tabular and text rows.
- {doc}`/ml/preparing/multimodal/decoding`: getting the bytes and turning them into tensors.
- {doc}`/api/accessors/media`: the full `.image` method list, with signatures.
