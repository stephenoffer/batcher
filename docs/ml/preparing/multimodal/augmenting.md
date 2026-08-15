# Transforming and fingerprinting images

This page covers the image operations that change pixels rather than shape them into a tensor: the geometry and colour transforms an augmentation policy is written from, and the perceptual hashes a deduplication pass is built on. All of them run in the Rust data plane over a binary column, so a corpus never leaves the engine to be turned.

{doc}`/ml/preparing/multimodal/decoding` covers getting bytes and decoding them; {doc}`/ml/preparing/multimodal/curating` covers the measures that decide which rows are worth keeping.

## Which output format an operation writes

Every operation that hands back an image takes a `format` and a `quality`. This matters more than it looks. A photographic corpus arrives as JPEG, and re-encoding it as PNG is both slower to write and several times larger, so a resize step meant to shrink a dataset inflates it instead:

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

`png` is the default, because it is lossless and it is what these operations wrote before the parameter existed. `quality` applies to the lossy containers only; the lossless ones have nothing to trade.

## Geometry

`rotate(degrees)` turns the image by a multiple of 90. Only right angles: a free rotation resamples every pixel and leaves a triangular border in a colour nobody chose, while a quarter turn is an exact transposition. Negative and over-full-turn values are normalized, so `-90` and `270` are the same rotation.

`flip_horizontal()` and `flip_vertical()` mirror an axis. The horizontal flip is the single most-used training-time augmentation, and it belongs on the *image* rather than in a loader so a detector's boxes can be flipped alongside it.

`pad(width, height, fill=0)` centres the image on a canvas without scaling it. That is the difference from `letterbox`, which fits and resamples: nothing here is resampled, so every surviving pixel keeps its exact value. An OCR or super-resolution pipeline needs that, and a canvas smaller than the image crops it centrally rather than failing the row.

```python
# docs: skip
from batcher import col

augmented = photos.with_columns(
    flipped=col("bytes").image.flip_horizontal(format="jpeg"),
    upright=col("bytes").image.rotate(-90),
    canvas=col("bytes").image.pad(640, 640, fill=114),
)
```

None of the three changes the channel count. An RGB image flipped stays RGB, which sounds obvious and is the thing a naive implementation gets wrong: the underlying helpers hand back RGBA whatever went in, so a flipped corpus silently grows a fourth channel and a third more bytes per row.

## Colour and tone

Four adjustments follow the `PIL.ImageEnhance` convention exactly, so an augmentation policy written against torchvision ports over unchanged. `1.0` is the identity and `0.0` the degenerate case:

| Method | `0.0` gives | What it varies |
|---|---|---|
| `adjust_brightness(factor)` | black | every channel scaled |
| `adjust_contrast(factor)` | a flat field at the image's own mean luma | spread about the mean |
| `adjust_saturation(factor)` | grayscale, still three channels | distance from grey |
| `adjust_hue(degrees)` | — (degrees wrap) | hue only, at constant saturation and value |

`blur(sigma)` and `sharpen(amount)` move detail rather than colour. `blur` is a Gaussian of `sigma` pixels; `sharpen` is the classical unsharp mask, `image + amount * (image - blur(image))`.

Four more come from the AutoAugment and RandAugment families. `posterize(bits)` keeps the top `bits` bits of each channel, masking rather than requantizing, so `bits=1` leaves only 0 and 128. `solarize(threshold)` inverts every channel value at or above the threshold. `invert()` is the photographic negative. `equalize()` flattens each channel's histogram, and `autocontrast(cutoff)` is the gentler alternative — it stretches the range without redistributing within it, ignoring `cutoff` percent of each tail:

```python
# docs: skip
from batcher import col

jittered = photos.select(
    warm=col("bytes").image.adjust_hue(15),
    vivid=col("bytes").image.adjust_saturation(1.4),
    fixed=col("bytes").image.autocontrast(cutoff=2.0),
)
```

`equalize()` and `autocontrast()` both leave a flat image alone rather than dividing by an empty range, which is how a solid-colour tile survives them instead of coming out as noise.

## Perceptual hashes and near-duplicate detection

A scraped corpus holds the same picture at three resolutions, two codecs and one watermark. None of that is findable by content hash — every byte differs — and finding it with a model costs an embedding per image. A perceptual hash costs a small decode and reduces the whole question to an integer comparison the engine already evaluates:

```python
# docs: skip
from batcher import col

fingerprinted = photos.with_columns(quick=col("bytes").image.ahash())
pairs = fingerprinted.join(fingerprinted.rename({"quick": "other"}), how="cross")
near_duplicates = pairs.filter(col("quick").bitwise_xor(col("other")).bit_count() <= 6)
```

Three exist because they trade the way every fingerprint family does:

| Method | How it reduces the image | Use it for |
|---|---|---|
| `ahash()` | 8x8 luma, thresholded at its own mean | the cheapest pre-filter |
| `dhash()` | 8x8 comparisons of horizontally adjacent pixels | a middle ground, robust to brightness shifts |
| `phash()` | the 8x8 lowest-frequency DCT coefficients of a 32x32 luma reduction, thresholded at their median | confirming a candidate — the most robust to rescaling, re-encoding and moderate cropping |

All three return an `Int64` whose bits *are* the hash, so `a.bitwise_xor(b).bit_count()` is the Hamming distance and a threshold on it is a similarity predicate. A dedup pass usually blocks on `ahash` and confirms with `phash`.

## Reading facts without decoding pixels

Four operations answer from the file header alone, which costs the bytes a decoder was going to touch anyway:

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

`format()` sniffs the magic bytes rather than the file extension, which is how a corpus full of `.jpg` files that are really PNGs gets found. It names the *container* — `jpeg`, not `jpg` — which is what `encode()` accepts and what a listing's own `format` column reports, so a value read off one can be handed to either of the others. Those rows decode fine and break whatever downstream step branched on the name. `aspect_ratio()` reports null rather than infinity for a zero-height image, so a filter written to find panoramas cannot silently accept it. `has_alpha()` decides whether a corpus needs flattening — `convert("RGB")` — before a model that takes three channels.

## Requirements and limitations

- `rotate` accepts multiples of 90 only. A free rotation is refused at plan build rather than resampled.
- WebP is readable but not writable, so it is not offered as an output `format`.
- `pad` flattens to RGB, because the canvas fill is a single byte value applied to all three channels.
- The hashes are stable across runs and machines: the luma reduction uses integer weights, so a stored hash stays comparable.

## See also

- {doc}`/ml/preparing/multimodal/curating`: the measures that decide which rows are worth keeping.
- {doc}`/ml/preparing/multimodal/decoding`: getting the bytes and turning them into tensors.
- {doc}`/api/relational/expression-accessors`: the full `.image` method list.
