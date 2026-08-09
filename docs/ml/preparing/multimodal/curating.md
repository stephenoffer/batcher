# Curating a corpus

A scraped corpus is mostly rows that decode perfectly and teach a model nothing. These are the screens that catch them.

## Curating an image corpus

A scraped image corpus is full of rows that decode perfectly and teach a model nothing: blank
placeholder tiles, all-white scans, out-of-focus photographs, and the grey box a CDN serves when
an asset is missing. None of them fails a decode, so nothing upstream catches them, and a vision
model trained on them learns the placeholder.

{py:meth}`.image.brightness() <batcher.plan.expr_ir.image._ImageNamespace.brightness>` is the blank detector. It reduces an image to its mean luma in `[0, 1]`,
and the useless rows sit at the extremes while a photograph of anything lands in the middle.

{py:meth}`.image.sharpness() <batcher.plan.expr_ir.image._ImageNamespace.sharpness>` is the focus measure, a normalized Laplacian variance. A sharp image has
strong second derivatives at its edges; a blurred or empty one has almost none.

```python
# docs: skip
from batcher import col, lit

brightness = col("bytes").image.brightness()
usable = photos.filter(
    (brightness > lit(0.05))          # not a black tile
    & (brightness < lit(0.95))        # not a blown-out scan
    & (col("bytes").image.sharpness() > lit(1e-4))  # not out of focus
)
```

Sharpness values are small in absolute terms — a well-focused photograph lands around 0.01 to
0.05 — so pick the threshold from a histogram of your own corpus rather than from a number you
read somewhere. It measures *detail*, not quality: a brick wall outscores a portrait, and a
noisy image outscores a clean one. Use it to find the blurred tail, not to rank images against
each other.

Both measure a downsampled copy, so the cost per image does not depend on its resolution. That
matters for sharpness specifically: at full resolution sensor noise reads as high-frequency
detail, and a blurry 50-megapixel photograph would score like a sharp one.

## Orienting photographs

A camera does not rotate its sensor data. It records which way up it was held in the EXIF
`Orientation` tag and stores the pixels as read, so a portrait phone photo is stored
landscape with a "rotate 90" note attached. Every viewer honours that note, and so does
`cv2.imread` and anything built on `PIL.ImageOps.exif_transpose`.

The decoder behind the {py:class}`.image <batcher.plan.expr_ir.image._ImageNamespace>` namespace does not. A corpus of phone photographs therefore
decodes a quarter turn from what the rest of your pipeline sees, and nothing about that is
visible: the decode succeeds, the tensor is the right shape, and the pixels are real. Two
things go wrong quietly. A model trains on sideways images, and any filter on shape selects
the wrong rows, because a portrait photo reports landscape dimensions.

`.image.exif_orientation()` says how much of a corpus is affected. `1` means upright, and
anything else means the stored pixels are not what a viewer shows:

```python
import batcher as bt
from batcher import col

photos = bt.from_pydict({"bytes": [b""]})
needs_rotation = photos.filter(col("bytes").image.exif_orientation() != 1)
```

`.image.auto_orient()` applies the transform, and composes in front of any other image
operation:

```python
# docs: skip
from batcher import col

upright = col("bytes").image.auto_orient()
tensors = photos.with_columns(x=upright.image.to_tensor(224, 224))
```

It is a separate operation rather than a changed default, because flipping the default
would rotate the output of pipelines that already compensate. The result is PNG, which
carries no EXIF, so the rotation cannot be applied twice by the next tool in the chain.

## Deduplicating images

Curating a scraped image corpus means dropping the same picture re-encoded, rescaled or
re-cropped. {py:meth}`.image.dhash() <batcher.plan.expr_ir.image._ImageNamespace.dhash>` is the primitive for it. It is a 64-bit *perceptual* hash
built from the gradients of a 9x8 grayscale thumbnail, so it survives re-encoding and
rescaling while still separating different pictures.

It returns a plain integer, so no new operator is needed. Exact-duplicate collapse is a
`distinct`, and near-duplicate matching is the Hamming distance you already have,
{py:meth}`bitwise_xor(...).bit_count() <batcher.plan.expr_ir.core.Expr.bitwise_xor>`, with a threshold of about 5 for "the same picture".

```python
# docs: skip
import batcher as bt
from batcher import col

photos = bt.read.images("s3://bucket/scrape/").with_columns(
    h=col("bytes").image.dhash()
)

# Exact duplicates: one row per distinct image.
unique = photos.distinct(subset=["h"])

# Near-duplicates against a reference set: a join plus a bit count.
pairs = (
    photos.select("uri", left=col("h"))
    .cross_join(reference.select(right=col("h")))
    .filter(col("left").bitwise_xor(col("right")).bit_count() <= 5)
)
```

A hash is null for an image that will not decode, so a corrupt file drops out of the
dedup rather than failing the pass.

## Cleaning scraped text

Scraped pages arrive as markup, and so do product descriptions and email bodies.
{py:meth}`.str.strip_html() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_html>` recovers the prose. It drops tags along with the contents of
`<script>` and `<style>`, strips comments and decodes entities, then collapses
whitespace, separating block elements with a space.

```python
import batcher as bt

pages = bt.from_pydict({"page": ["<p>Tom &amp; Jerry</p><p>x</p><script>f()</script>"]})
print(pages.select(text=bt.col("page").str.strip_html()).to_pydict())
# {'text': ['Tom & Jerry x']}
```

Reach for this over the {py:meth}`regexp_replace('<[^>]*>', '') <batcher.plan.expr_ir.namespaces.strings._StrNamespace.regexp_replace>` idiom, which is wrong in three
ways that quietly poison a corpus. It leaves the JavaScript in `<script>` as prose, it
leaves `&amp;` and `&nbsp;` undecoded, and it welds `<p>a</p><p>b</p>` into `ab`.
{py:meth}`.str.strip_html() <batcher.plan.expr_ir.namespaces.strings._StrNamespace.strip_html>` is a text extractor, not an HTML parser, so malformed markup never
raises and one bad row in a web scrape cannot abort the scan.
