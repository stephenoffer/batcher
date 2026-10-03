# Multimodal data

This section covers turning media into columns a model can read: fetching the bytes, decoding them into tensors, curating what comes back, and moving the result through a pipeline without paying for it twice.

References become bytes, bytes become tensors, and tensors reach a model. Each link is a lazy operator over whole batches, and the decode is an expression on the `.image`, `.audio` or `.video` namespace, implemented in Rust. A million images never turn into a million Python calls, and the same expressions work as filters.

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

Swap {py:obj}`bt.from_pydict <batcher.from_pydict>` for `bt.read.images("s3://<your-bucket>/images/")` or `ds.ml.download("url")` over a table of links, and the rest of the plan is unchanged.

The following diagram traces the two rows through that plan:

![Two PNGs, id 1 noisy and id 2 black, enter one plan that runs on one collect. The screen, brightness() greater than 0.05, keeps id 1 and drops id 2, so the black image never reaches the tensor or the hash. The surviving row goes to two expressions in the same pass: to_tensor(8, 8) decodes it into an 8 by 8 by 3 uint8 tensor in the image column, and phash() fingerprints it as a 64-bit hash in an Int64 phash column. The output is one row with id, image and phash. Each step is a Rust expression over whole batches, so no image becomes a Python call.](/_static/diagrams/media_screen_pass.svg)

## Audio works the same way

The `.audio` measures reduce a clip to a number, so screening a corpus is a predicate. The example below writes two one-second WAV clips with the standard library, a tone and silence:

```python
import wave


def wav(samples, rate=16000):
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes((samples * 32767).astype("<i2").tobytes())
    return buf.getvalue()


t = np.arange(16000) / 16000
clips = bt.from_pydict(
    {"id": [1, 2], "bytes": [wav(0.5 * np.sin(2 * np.pi * 440 * t)), wav(np.zeros(16000))]}
)
print(
    clips.select(
        "id",
        level=col("bytes").audio.rms().round(3),
        silent=col("bytes").audio.silence_ratio().round(2),
    ).to_pydict()
)
# {'id': [1, 2], 'level': [0.354, 0.0], 'silent': [0.02, 1.0]}
```

Keep only the clips with signal:

```python
print(clips.filter(col("bytes").audio.rms() > 0.01).select("id").to_pydict())
# {'id': [1]}
```

## What runs natively

The following table summarizes the in-engine surface per modality. {doc}`/api/accessors/media` lists every method, with signatures:

| Modality | Decode and shape | Measure and curate |
|---|---|---|
| Image | `decode`, `to_tensor`, `to_tensor_f32` with mean and std normalization, `resize`, `crop`, `center_crop`, `letterbox`, `rotate`, `flip_horizontal`, photometric adjustments, `encode` | `brightness`, `sharpness`, `entropy`, `colorfulness`, `phash`, `dhash`, `ahash`, `aspect_ratio`, `has_alpha` |
| Audio | `decode`, `to_waveform`, `resample`, `slice`, `pad_or_trim`, `mel_spectrogram`, `mfcc`, `encode_wav` | `rms`, `dbfs`, `peak_dbfs`, `clipping_ratio`, `silence_ratio`, spectral centroid, rolloff, bandwidth, and flatness |
| Video | `decode`, `frames`, `frame_at`, `thumbnail` | Header metadata from the reader, before any frame decodes |

The mel spectrogram and MFCC match `torchaudio` to 1e-6, and the image photometric operations follow `PIL.ImageEnhance` and AutoAugment conventions, so a torchvision augmentation policy ports without retuning.

## How fast it is

Decoding and resizing 2,000 JPEGs to 224x224 on one 96-core machine runs at 4,649-4,788 img/s, 1.87-1.96x Daft and 6.35-6.61x Ray Data on the same frames. {doc}`/benchmarks/results/multimodal-ingest` has the details.

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
- {doc}`/examples/multimodal`: image, audio, video, and text-analytics scripts.

```{toctree}
:hidden:

decoding
augmenting
curating
audio
video
pipelines
```
