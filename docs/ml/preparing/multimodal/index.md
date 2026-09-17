# Multimodal data

This section covers turning media into columns a model can read: fetching the bytes, decoding them into tensors, curating what comes back, and moving the result through a pipeline without paying for it twice.

The chain is always the same. References such as URLs and file paths become bytes, bytes become tensors, and tensors reach a model. In Batcher each link is a lazy operator over whole batches. The decode itself is an expression on the `.image`, `.audio` or `.video` namespace, implemented in Rust, so a million images never turn into a million Python calls. The same expressions work as filters, which is how you drop a blank or blurred image before anything expensive touches it.

## Decode, screen, and fingerprint in one pass

The example below builds two PNGs in memory, one noisy and one black. It drops the black one with a brightness predicate, decodes the survivor to an 8x8 tensor, and computes a perceptual hash for deduplication, all in one plan:

```python
import io

import numpy as np
from PIL import Image

import batcher as bt
from batcher import col


def png(pixels):
    buf = io.BytesIO()
    Image.fromarray(pixels).save(buf, format="PNG")
    return buf.getvalue()


noisy = (np.random.default_rng(0).random((10, 12, 3)) * 255).astype("uint8")
black = np.zeros((10, 12, 3), dtype="uint8")
ds = bt.from_pydict({"id": [1, 2], "bytes": [png(noisy), png(black)]})

out = (
    ds.filter(col("bytes").image.brightness() > 0.05)
    .with_columns(image=col("bytes").image.to_tensor(8, 8), phash=col("bytes").image.phash())
    .select("id", "image", "phash")
    .collect()
)
print(out.column("id").to_pylist(), out.schema.field("image").type)
# [1] extension<arrow.fixed_shape_tensor[value_type=uint8, shape=[8,8,3]]>
```

Swap `bt.from_pydict` for `bt.read.images("s3://<your-bucket>/images/")` or `ds.ml.download("url")` over a table of links, and the rest of the plan is unchanged.

The following diagram traces the two rows through that plan:

![Two PNGs, id 1 noisy and id 2 black, enter one plan that runs on one collect. The screen, brightness() greater than 0.05, keeps id 1 and drops id 2, so the black image never reaches the tensor or the hash. The surviving row goes to two expressions in the same pass: to_tensor(8, 8) decodes it into an 8 by 8 by 3 uint8 tensor in the image column, and phash() fingerprints it as a 64-bit hash in an Int64 phash column. The output is one row with id, image and phash. Each step is a Rust expression over whole batches, so no image becomes a Python call.](/_static/diagrams/media_screen_pass.svg)

## What runs natively

The following table summarizes the in-engine surface per modality. {doc}`/api/relational/expression-accessors` lists every method:

| Modality | Decode and shape | Measure and curate |
|---|---|---|
| Image | `decode`, `to_tensor`, `to_tensor_f32` with mean and std normalization, `resize`, `crop`, `center_crop`, `letterbox`, `rotate`, `flip_horizontal`, photometric adjustments, `encode` | `brightness`, `sharpness`, `entropy`, `colorfulness`, `phash`, `dhash`, `ahash`, `aspect_ratio`, `has_alpha` |
| Audio | `decode`, `to_waveform`, `resample`, `slice`, `pad_or_trim`, `mel_spectrogram`, `mfcc`, `encode_wav` | `rms`, `dbfs`, `peak_dbfs`, `clipping_ratio`, `silence_ratio`, spectral centroid, rolloff, bandwidth, and flatness |
| Video | `decode`, `frames`, `frame_at`, `thumbnail` | Header metadata from the reader, before any frame decodes |

The mel spectrogram and MFCC match `torchaudio` to 1e-6, and the image photometric operations follow `PIL.ImageEnhance` and AutoAugment conventions, so a torchvision augmentation policy ports without retuning.

## How fast it is

On one busy 96-core machine with a release build, decoding and resizing 2,000 JPEGs to 224x224 ran at 4,649-4,788 img/s, and at 5,693 img/s in a separate run on an idle 96-core node. That is 1.87-1.96x Daft 0.7.23 and 6.35-6.61x Ray Data 2.56 on the same frames, with every engine required to return the same frame count at the same shape first. Computing entropy, a perceptual hash and a horizontal flip per image ran 5.67-5.76x faster than the per-row Pillow loop a Daft or Ray Data user would write for those, and decoding 2,000 audio clips took 20.4 ms, 19.8x a per-clip `soundfile` loop. {doc}`/benchmarks/results/multimodal-ingest` has the details.

## In this section

The following table lists the pages in the order a new media pipeline usually needs them:

| Page | Covers |
|---|---|
| {doc}`/ml/preparing/multimodal/decoding` | Fetching remote bytes, decoding into tensors, resize modes, bad rows, and tensor columns of mixed shapes. |
| {doc}`/ml/preparing/multimodal/augmenting` | Geometry and color transforms, perceptual hashes, and facts read without decoding pixels. |
| {doc}`/ml/preparing/multimodal/curating` | Screening a scraped image corpus, orienting photographs, deduplicating, and cleaning scraped text. |
| {doc}`/ml/preparing/multimodal/audio` | Measuring recording quality, leveling clips, writing audio back out, and spectral features. |
| {doc}`/ml/preparing/multimodal/video` | Reading metadata first, sampling frames, pulling stills, and which decoder is running. |
| {doc}`/ml/preparing/multimodal/pipelines` | Keeping large payloads out of shuffles, the path from references to predictions, and vector search over the result. |

## Requirements and limitations

- The readers take the `multimodal` extra, or `image`, `audio` and `video` individually: `pip install 'batcher-engine[multimodal]'`.
- Native audio decode covers WAV/PCM and FLAC. Other containers decode to null.
- Native video decode needs an engine built with the `video` cargo feature against FFmpeg. Without it, video falls back to Python decoding through the `video` extra.

## See also

- {doc}`/ml/inference/inference`: run a model over the decoded tensors.
- {doc}`/ml/retrieval/embeddings`: turn images or text into vectors.
- {doc}`/ml/preparing/preprocessors/index`: assemble decoded features into a training matrix.
- {doc}`/getting-started/tutorials/ml/batch-inference`: a scoring pipeline built step by step.

```{toctree}
:hidden:

decoding
augmenting
curating
audio
video
pipelines
```
