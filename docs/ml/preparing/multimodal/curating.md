# Curating a corpus

This page covers the screens that catch rows which decode perfectly and teach a model nothing. Much of a scraped image corpus is exactly that, and scraped text has its own version of the problem.

## Screening an image corpus

Blank placeholder tiles, all-white scans, out-of-focus photographs, the gray box a CDN serves when an asset is missing: none of them fails a decode. Nothing upstream catches them, and a vision model trained on them learns the placeholder.

{py:meth}`.image.brightness() <batcher.plan.expr_ir.image._ImageNamespace.brightness>` is the blank detector. It reduces an image to its mean luma in `[0, 1]`. The useless rows sit at the extremes, and a photograph of anything lands in the middle. {py:meth}`.image.sharpness() <batcher.plan.expr_ir.image._ImageNamespace.sharpness>` is the focus measure, a normalized Laplacian variance. A sharp image has strong second derivatives at its edges, and a blurred or empty one has almost none.

```python
# docs: skip
from batcher import col, lit

brightness = col("bytes").image.brightness()
usable = photos.filter(
    (brightness > lit(0.05))  # not a black tile
    & (brightness < lit(0.95))  # not a blown-out scan
    & (col("bytes").image.sharpness() > lit(1e-4))  # not out of focus
)
```

Sharpness values are small. A well-focused photograph lands around 0.01 to 0.05, so pick the threshold from a histogram of your own corpus. The measure reads *detail*, not quality: a brick wall outscores a portrait, and a noisy image outscores a clean one. Use it to find the blurred tail, never to rank images against each other.

Both measures read a downsampled copy, so the cost per image doesn't depend on resolution. That matters most for sharpness. At full resolution, sensor noise reads as high-frequency detail, and a blurry 50-megapixel photograph would score like a sharp one.

## What the listing knows before anything decodes

`bt.read.images` emits `width`, `height`, `mode` and `format` per file, read from the header in the same pass that fetched the bytes. `format` is the container the bytes actually are, and it earns its place beside `mime` for the same reason {py:meth}`.image.format() <batcher.plan.expr_ir.image._ImageNamespace.format>` does. A corpus assembled by content type is full of files whose extension and container disagree. Those rows decode fine and break whatever downstream step branched on the name.

The listing matches on file extension, so a format the source doesn't name is invisible. The read returns nothing, and the result looks like an empty directory rather than an unlisted format. That's why `.heic`, `.avif`, `.jfif`, `.jp2` and similar extensions are listed even where Pillow needs a plugin to decode them. The rows are still worth having: `bytes`, `size` and `mime` come from the read itself, and an unparseable header nulls that file's metadata columns without dropping its row.

## The rows a luma measure can't see

Brightness and sharpness both read the gray channel, so three kinds of useless row slip past them. Each has its own measure. All of them read the same downsampled copy, so adding them to a filter costs nothing beyond the decode you already pay for.

{py:meth}`.image.entropy() <batcher.plan.expr_ir.image._ImageNamespace.entropy>` is the Shannon entropy of the luma histogram, in bits. It separates what brightness can't: a mid-gray placeholder tile and a photograph of a foggy road have the same mean and completely different information content. A solid field scores 0 whatever its shade, a two-tone logo near 1, and a photograph of anything between 6 and 8.

{py:meth}`.image.colorfulness() <batcher.plan.expr_ir.image._ImageNamespace.colorfulness>` is the Hasler-Süsstrunk metric. A sepia-toned duplicate, a line drawing and a scanned page all have ordinary brightness, sharpness and entropy, and all are the wrong training data for a model meant to see color. Anything gray scores roughly 0, and a vivid scene 15 or more.

{py:meth}`.image.is_grayscale() <batcher.plan.expr_ir.image._ImageNamespace.is_grayscale>` finds grayscale images *stored* as three identical channels. No header reports them. `decode()` says `RGB`, `has_alpha()` says false, and nothing says that two thirds of every tensor is a copy. Once you find them, you can route them to a one-channel model instead of paying three times the bandwidth for one channel of information.

{py:meth}`.image.mean_color() <batcher.plan.expr_ir.image._ImageNamespace.mean_color>` reports the three channel means as a struct. It's the cheapest color summary there is, and it turns "every product shot on a white background" or "cluster this corpus by palette" into ordinary expressions instead of an embedding model.

```python
# docs: skip
from batcher import col, lit

background = col("bytes").image.mean_color()
usable = photos.filter(
    (col("bytes").image.entropy() > lit(4.0))  # not a placeholder tile
    & (col("bytes").image.colorfulness() > lit(5.0))  # not a scan or a line drawing
    & ~col("bytes").image.is_grayscale()  # not gray stored as RGB
)
on_white = photos.filter(background.struct.field("r") > lit(240.0))
```

## Orienting photographs

A camera doesn't rotate its sensor data. It records which way up it was held in the EXIF `Orientation` tag and stores the pixels as read, so a portrait phone photo is stored landscape with a "rotate 90" note attached. Every viewer honors that note, and so do `cv2.imread` and anything built on `PIL.ImageOps.exif_transpose`.

The decoder behind the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` namespace doesn't. A corpus of phone photographs therefore decodes a quarter turn from what the rest of your pipeline sees, and nothing shows it. The decode succeeds, the tensor has the right shape, and the pixels are real. Two things go wrong quietly: a model trains on sideways images, and any filter on shape selects the wrong rows, because a portrait photo reports landscape dimensions.

`.image.exif_orientation()` tells you how much of a corpus is affected. `1` means upright, and anything else means the stored pixels aren't what a viewer shows:

```python
import batcher as bt
from batcher import col

photos = bt.from_pydict({"bytes": [b""]})
needs_rotation = photos.filter(col("bytes").image.exif_orientation() != 1)
```

`.image.auto_orient()` applies the transform, and it composes in front of any other image operation:

```python
# docs: skip
from batcher import col

upright = col("bytes").image.auto_orient()
tensors = photos.with_columns(x=upright.image.to_tensor(224, 224))
```

It's a separate operation rather than a changed default, because a new default would rotate the output of pipelines that already compensate. The output is PNG unless you pass `format`. PNG carries no EXIF, so the next tool in the chain can't apply the rotation a second time.

## Deduplicating images

A scraped image corpus holds the same picture re-encoded, rescaled or re-cropped. {py:meth}`.image.dhash() <batcher.plan.expr_ir.image._ImageNamespace.dhash>` is the primitive for dropping those copies. It's a 64-bit *perceptual* hash built from the gradients of a 9x8 grayscale thumbnail, so it survives re-encoding and rescaling while still separating different pictures.

The hash is a plain integer, so no new operator is needed. Exact-duplicate collapse is a `distinct`. Near-duplicate matching is a Hamming distance, {py:meth}`bitwise_xor(...).bit_count() <batcher.plan.expr_ir.core.Expr.bitwise_xor>`, with a threshold of about 5 for "the same picture".

```python
# docs: skip
import batcher as bt
from batcher import col

photos = bt.read.images("s3://bucket/scrape/").with_columns(h=col("bytes").image.dhash())

# Exact duplicates: one row per distinct image.
unique = photos.distinct(subset=["h"])

# Near-duplicates against a reference set: a join plus a bit count.
pairs = (
    photos.select("uri", left=col("h"))
    .cross_join(reference.select(right=col("h")))
    .filter(col("left").bitwise_xor(col("right")).bit_count() <= 5)
)
```

A hash is null for an image that won't decode, so a corrupt file drops out of the dedup instead of failing the pass. {doc}`/ml/preparing/multimodal/augmenting` compares `dhash` with `ahash` and `phash`, which trade cost against robustness.

## Cleaning scraped text

Scraped pages arrive as markup, and so do product descriptions and email bodies. {py:meth}`.str.strip_html() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_html>` recovers the prose. It drops tags along with the contents of `<script>` and `<style>`, strips comments, decodes entities, and collapses whitespace, separating block elements with a space.

```python
import batcher as bt

pages = bt.from_pydict({"page": ["<p>Tom &amp; Jerry</p><p>x</p><script>f()</script>"]})
print(pages.select(text=bt.col("page").str.strip_html()).to_pydict())
# {'text': ['Tom & Jerry x']}
```

Prefer it to the {py:meth}`regexp_replace('<[^>]*>', '') <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_replace>` idiom, which quietly poisons a corpus three ways. It leaves the JavaScript in `<script>` as prose, it leaves `&amp;` and `&nbsp;` undecoded, and it welds `<p>a</p><p>b</p>` into `ab`. `strip_html` is a text extractor, not an HTML parser, so malformed markup never raises and one bad row in a web scrape can't abort the scan.

## See also

- {doc}`/ml/preparing/multimodal/decoding`: fetching bytes and decoding them into tensors.
- {doc}`/ml/preparing/multimodal/augmenting`: geometry and color transforms, and the three perceptual hashes.
- {doc}`/ml/preparing/multimodal/audio`: the same triage for recording quality in an audio corpus.
- {doc}`/ml/preparing/preprocessors/deduplication`: deduplicating tabular and text rows.
- {doc}`/api/relational/expression-accessors`: the full `.image` and `.str` method lists.
